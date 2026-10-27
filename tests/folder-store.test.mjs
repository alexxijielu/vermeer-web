import test from 'node:test';
import assert from 'node:assert/strict';
import { ModelStore, FolderModelStore, downloadModelFolder } from '../web/model-store.js';
import { MemoryDirectory } from './memory-directory.mjs';

async function fixture() {
  const data = {
    'esm.onnx': Uint8Array.of(1, 2, 3),
    'decoder.onnx': Uint8Array.of(4, 5, 6),
    'b.onnx': Uint8Array.of(7, 8, 9),
    'b-weights/0.bin': Uint8Array.of(10, 11, 12),
    'l.onnx': Uint8Array.of(13, 14, 15),
  };
  const files = {};
  for (const [name, bytes] of Object.entries(data)) {
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    files[name] = { bytes: bytes.length, sha256: Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('') };
  }
  return {
    data, directory: new MemoryDirectory(),
    manifest: { files, commonModels: ['esm.onnx', 'decoder.onnx'], variants: {
      B: { graph: 'b.onnx', externalData: ['b-weights/0.bin'] },
      L: { graph: 'l.onnx', externalData: [] },
    } },
  };
}

test('folder download is complete, hash-qualified, resumable and reuses shared files across variants', async () => {
  const { data, directory, manifest } = await fixture();
  const requests = [];
  const fetcher = async (url, options) => {
    assert.equal(options.cache, 'no-store');
    const name = new URL(url).pathname.slice('/assets/models/'.length);
    assert.equal(new URL(url).searchParams.get('sha256'), manifest.files[name].sha256);
    requests.push(name);
    return new Response(data[name]);
  };
  const progress = [];
  const info = await downloadModelFolder(manifest, 'B', directory, (done, total) => progress.push([done, total]), fetcher);
  assert.deepEqual(requests, ['esm.onnx', 'decoder.onnx', 'b.onnx', 'b-weights/0.bin']);
  assert.deepEqual(info, { totalBytes: 12, files: 4 });
  assert.deepEqual(progress.at(-1), [12, 12]);
  requests.length = 0;
  await downloadModelFolder(manifest, 'B', directory, () => {}, fetcher);
  assert.deepEqual(requests, []);
  const freshVisit = new FolderModelStore(manifest, directory);
  freshVisit.fetcher = () => { throw new Error('Local reads must not fetch'); };
  Object.defineProperty(freshVisit, 'cache', { get() { throw new Error('Local reads must not access Cache Storage'); } });
  for (const name of ['esm.onnx', 'decoder.onnx', 'b.onnx', 'b-weights/0.bin']) {
    await freshVisit.prepare(name);
    assert.deepEqual(await freshVisit.read(name), data[name]);
  }
  await downloadModelFolder(manifest, 'L', directory, () => {}, fetcher);
  assert.deepEqual(requests, ['l.onnx']);
  assert.deepEqual(await freshVisit.read('b.onnx'), data['b.onnx']);
});

test('incomplete downloads resume and corrupt existing files are replaced only with verified bytes', async () => {
  const { data, directory, manifest } = await fixture();
  let fail = true;
  const requests = [];
  const fetcher = async url => {
    const name = new URL(url).pathname.slice('/assets/models/'.length);
    requests.push(name);
    return fail && name === 'decoder.onnx' ? new Response('', { status: 503 }) : new Response(data[name]);
  };
  await assert.rejects(downloadModelFolder(manifest, 'B', directory, () => {}, fetcher), /HTTP 503/);
  assert.deepEqual(directory.files.get('esm.onnx'), data['esm.onnx']);
  fail = false;
  requests.length = 0;
  await downloadModelFolder(manifest, 'B', directory, () => {}, fetcher);
  assert.deepEqual(requests, ['decoder.onnx', 'b.onnx', 'b-weights/0.bin']);
  directory.files.set('b.onnx', Uint8Array.of(0, 0, 0));
  await assert.rejects(downloadModelFolder(manifest, 'B', directory, () => {}, async () => new Response(Uint8Array.of(9, 9, 9))), /integrity/);
  assert.deepEqual(directory.files.get('b.onnx'), Uint8Array.of(0, 0, 0));
  requests.length = 0;
  await downloadModelFolder(manifest, 'B', directory, () => {}, fetcher);
  assert.deepEqual(requests, ['b.onnx']);
});

test('model downloads use pinned external hosting and include verified license notices', async () => {
  const { directory, manifest, data } = await fixture();
  manifest.downloadBaseUrl = `https://huggingface.co/example/models/resolve/${'a'.repeat(40)}/`;
  manifest.notices = ['licenses/model.txt'];
  manifest.files['licenses/model.txt'] = manifest.files['esm.onnx'];
  const requests = [];
  await downloadModelFolder(manifest, 'B', directory, () => {}, async (url, options) => {
    assert.ok(url.startsWith(manifest.downloadBaseUrl));
    assert.equal(options.cache, 'no-store');
    const name = new URL(url).pathname.slice(new URL(manifest.downloadBaseUrl).pathname.length);
    requests.push(name);
    return new Response(name === 'licenses/model.txt' ? data['esm.onnx'] : data[name]);
  });
  assert.equal(requests.at(-1), 'licenses/model.txt');
  const store = new FolderModelStore(manifest, directory);
  store.fetcher = () => { throw new Error('Local inference must not use the external host'); };
  await store.prepare('b.onnx');
  assert.deepEqual(await store.read('b.onnx'), data['b.onnx']);
});

test('default asset URLs respect a project-site subpath and custom-domain root', async () => {
  const { manifest } = await fixture();
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'location');
  try {
    for (const prefix of ['https://example.github.io/vermeer-web/', 'https://vermeer.example.com/']) {
      Object.defineProperty(globalThis, 'location', { configurable: true, value: { href: `${prefix}#results` } });
      assert.equal(new ModelStore(manifest, null).key('esm.onnx'), `${prefix}assets/models/esm.onnx?sha256=${manifest.files['esm.onnx'].sha256}`);
      assert.equal(new ModelStore(manifest, null, fetch, { basePath: './assets/' }).key('esm.onnx'), `${prefix}assets/esm.onnx?sha256=${manifest.files['esm.onnx'].sha256}`);
    }
  } finally {
    if (previous) Object.defineProperty(globalThis, 'location', previous);
    else delete globalThis.location;
  }
});

test('local setup and prediction reject missing, modified or unprepared files without fetching', async () => {
  const { data, directory, manifest } = await fixture();
  const store = new FolderModelStore(manifest, directory);
  store.fetcher = () => { throw new Error('No network fallback'); };
  await assert.rejects(store.read('b.onnx'), /not prepared/);
  await assert.rejects(store.prepare('b.onnx'), /Missing local model file/);
  directory.files.set('b.onnx', data['b.onnx']);
  await store.prepare('b.onnx');
  directory.files.set('b.onnx', Uint8Array.of(0));
  await assert.rejects(store.read('b.onnx'), /Incomplete model file/);
  directory.files.set('b.onnx', Uint8Array.of(0, 0, 0));
  await assert.rejects(store.read('b.onnx'), /integrity/);
  directory.files.delete('b.onnx');
  await assert.rejects(store.read('b.onnx'), /Missing local model file/);
});

test('folder permission failures and invalid paths are explicit, never treated as missing downloads', async () => {
  const { directory, manifest } = await fixture();
  directory.getFileHandle = async () => { throw new DOMException('Denied', 'NotAllowedError'); };
  const store = new FolderModelStore(manifest, directory);
  await assert.rejects(store.prepare('b.onnx'), /Permission to read/);
  await assert.rejects(downloadModelFolder(manifest, 'B', directory, () => {}, async () => { throw new Error('Must not fetch'); }), /Permission to read/);
  manifest.files['../outside'] = manifest.files['b.onnx'];
  await assert.rejects(store.prepare('../outside'), /Invalid model path/);
});

test('failed writes abort temporary data and preserve both write and cleanup errors', async () => {
  const { directory, manifest, data } = await fixture();
  const original = directory.getFileHandle.bind(directory);
  const writeError = new Error('Disk full');
  const cleanupError = new Error('Stream already closed');
  let aborted = false;
  directory.getFileHandle = async (...args) => {
    const handle = await original(...args);
    handle.createWritable = async () => ({
      write: async () => { throw writeError; },
      abort: async () => { aborted = true; throw cleanupError; },
    });
    return handle;
  };
  await assert.rejects(downloadModelFolder(manifest, 'B', directory, () => {}, async () => new Response(data['esm.onnx'])), error => {
    assert.ok(error instanceof AggregateError);
    assert.deepEqual(error.errors, [writeError, cleanupError]);
    assert.match(error.message, /Disk full/);
    assert.match(error.message, /Stream already closed/);
    return true;
  });
  assert.equal(aborted, true);
  assert.equal(directory.files.get('esm.onnx').length, 0);
});

test('incorrect local file sizes are rejected before allocating their contents', async () => {
  const { manifest } = await fixture();
  const directory = {
    getFileHandle: async () => ({
      getFile: async () => ({
        size: 1e12,
        arrayBuffer: async () => { throw new Error('Must not allocate a huge invalid file'); },
      }),
    }),
  };
  await assert.rejects(new FolderModelStore(manifest, directory).prepare('b.onnx'), /Incomplete model file/);
});

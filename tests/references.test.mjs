import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ReferenceStore, downloadReferenceFolder, referenceFiles, selectableCellLines } from '../web/reference-store.js';
import { MemoryDirectory } from './memory-directory.mjs';
import { downloadLocalBundle } from '../web/local-bundle.js';

test('the selector excludes only Sperm and serum-starved RPE1 without changing the compatible reference pack', async () => {
  const catalog = JSON.parse(await readFile(new URL('../web/assets/references.json', import.meta.url), 'utf8'));
  const original = Object.keys(catalog.cellLines);
  const selected = selectableCellLines(catalog);
  assert.equal(selected.length, 39);
  assert.ok(selected.includes('hTERT-RPE1'));
  assert.deepEqual(original.filter(name => !selected.includes(name)), ['hTERT-RPE1 (serum starved)', 'Sperm']);
  assert.deepEqual(Object.keys(catalog.cellLines), original);
  assert.deepEqual(selectableCellLines({ cellLines: { Rh30: [], 'hTERT-RPE1': [] } }), ['Rh30', 'hTERT-RPE1']);
  assert.deepEqual(selectableCellLines({ cellLines: { Sperm: [], 'hTERT-RPE1 (serum starved)': [] } }), []);
});

async function fixture() {
  const catalog = { version: 2, fieldsPerCellLine: 10, cellLines: {} };
  const data = {};
  const prefix = [0, 1, 2].flatMap(channel => [16384 + channel, ...Array(256).fill(channel), 16388 + channel]);
  for (const [line, slug] of [['Rh30', 'Rh30'], ['HUVEC/TERT2', 'HUVEC_TERT2']]) {
    catalog.cellLines[line] = [];
    for (let index = 1; index <= 10; index++) {
      const path = `references/${slug}/${String(index).padStart(2, '0')}`;
      const id = `${slug}-${index}`;
      const images = ['reference.png', 'landmark-nucleus.png', 'landmark-microtubules.png', 'landmark-er.png'].map(name => `${path}/${name}`);
      data[`${path}.json`] = new TextEncoder().encode(JSON.stringify({ id, index, cellLine: line, prefix, images }));
      images.forEach((name, channel) => { data[name] = Uint8Array.of(index, channel, 42); });
      catalog.cellLines[line].push({ index, id, url: `/assets/${path}.json` });
    }
  }
  data['references.json'] = new TextEncoder().encode(JSON.stringify(catalog));
  const files = {};
  for (const [name, bytes] of Object.entries(data)) {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    files[name] = { bytes: bytes.length, sha256: Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('') };
  }
  const manifest = { version: 1, cellLineCount: 2, fieldCount: 20, files, browserFiles: Object.keys(files) };
  const requests = [];
  const fetcher = async (url, options) => {
    assert.equal(options.cache, 'no-store');
    const name = new URL(url).pathname.slice('/assets/'.length);
    assert.equal(new URL(url).searchParams.get('sha256'), files[name].sha256);
    requests.push(name);
    return new Response(data[name]);
  };
  return { manifest, data, catalog, requests, fetcher, directory: new MemoryDirectory() };
}

test('catalog and selected reference load only one field, never the whole image collection or TIFFs', async () => {
  const { manifest, fetcher, requests } = await fixture();
  const store = new ReferenceStore(manifest, null, fetcher);
  const catalog = await store.catalog();
  assert.deepEqual(requests, ['references.json']);
  requests.length = 0;
  const { reference, images } = await store.load(catalog.cellLines['HUVEC/TERT2'][9], 'HUVEC/TERT2');
  assert.equal(reference.index, 10);
  assert.equal(reference.prefix.length, 774);
  assert.equal(images.size, 4);
  assert.equal(requests.length, 5);
  assert.ok(requests.every(name => name.startsWith('references/HUVEC_TERT2/10')));
  assert.equal(Object.hasOwn(store, 'imageCache'), false);
});

test('local reference packs resume, read directly without network or Cache Storage, and report missing files', async () => {
  const { manifest, directory, fetcher, requests } = await fixture();
  const first = await downloadReferenceFolder(manifest, directory, () => {}, fetcher);
  assert.equal(first.files, manifest.browserFiles.length);
  assert.ok(requests.every(name => name.endsWith('.png') || name.endsWith('.json')));
  requests.length = 0;
  await downloadReferenceFolder(manifest, directory, () => {}, fetcher);
  assert.deepEqual(requests, []);
  const store = new ReferenceStore(manifest, directory, () => { throw new Error('Local references cannot fetch'); });
  Object.defineProperty(store.local, 'cache', { get() { throw new Error('No Cache Storage copies'); } });
  const catalog = await store.catalog();
  const selected = catalog.cellLines.Rh30[0];
  const loaded = await store.load(selected, 'Rh30');
  assert.equal(loaded.images.size, 4);
  const folder = await (await directory.getDirectoryHandle('references')).getDirectoryHandle('Rh30');
  folder.files.delete('01.json');
  await assert.rejects(store.load(selected, 'Rh30'), /Missing local reference file/);
  assert.deepEqual(requests, []);
});

test('cancelled reference requests do not fetch or complete after source changes', async () => {
  const { manifest, fetcher, requests, catalog } = await fixture();
  const controller = new AbortController();
  controller.abort();
  const store = new ReferenceStore(manifest, null, fetcher);
  await assert.rejects(store.load(catalog.cellLines.Rh30[0], 'Rh30', controller.signal), { name: 'AbortError' });
  assert.equal(requests.length, 0);
  const lateController = new AbortController();
  const lateStore = new ReferenceStore(manifest, null, async (...args) => {
    const response = await fetcher(...args);
    lateController.abort();
    return response;
  });
  await assert.rejects(lateStore.catalog(lateController.signal), { name: 'AbortError' });
});

test('corrupt reference responses and incorrect cell identity produce explicit errors', async () => {
  const { manifest, catalog, fetcher } = await fixture();
  const corrupt = new ReferenceStore(manifest, null, async () => new Response(Uint8Array.of(1)));
  await assert.rejects(corrupt.catalog(), /Incomplete reference file/);
  const store = new ReferenceStore(manifest, null, fetcher);
  await assert.rejects(store.load(catalog.cellLines.Rh30[0], 'HUVEC/TERT2'), /Invalid reference field/);
  const failing = new ReferenceStore(manifest, null, async () => new Response('', { status: 404 }));
  await assert.rejects(failing.catalog(), /HTTP 404/);
});

test('reference manifests reject raw TIFFs and unsafe paths rather than downloading them', async () => {
  const { manifest } = await fixture();
  assert.equal(referenceFiles(manifest).length, 101);
  for (const name of ['reference-tiffs/field.tif.gz', '../outside.json', 'references/Rh30/../unsafe.json']) {
    const changed = { ...manifest, files: { ...manifest.files, [name]: { bytes: 1, sha256: '0'.repeat(64) } }, browserFiles: [...manifest.browserFiles, name] };
    assert.throws(() => referenceFiles(changed), /Only prepared PNGs and JSON/);
  }
});

test('one local bundle downloads models and all references, reusing existing files with combined progress', async () => {
  const { manifest, directory, fetcher, requests } = await fixture();
  const bytes = Uint8Array.of(7, 8, 9);
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
  const models = { commonModels: ['esm.onnx'], variants: { B: { graph: 'b.onnx', externalData: [] } }, files: {
    'esm.onnx': { bytes: 3, sha256: hash }, 'b.onnx': { bytes: 3, sha256: hash },
  } };
  const allRequests = [];
  const combinedFetch = async (url, options) => {
    allRequests.push(url);
    if (new URL(url).pathname.startsWith('/assets/models/')) {
      assert.equal(options.cache, 'no-store');
      return new Response(bytes);
    }
    return fetcher(url, options);
  };
  const progress = [];
  const info = await downloadLocalBundle(models, manifest, 'B', directory, (done, total) => progress.push([done, total]), combinedFetch);
  assert.equal(info.files, 103);
  assert.deepEqual(progress.at(-1), [info.totalBytes, info.totalBytes]);
  assert.ok(progress.every(([done, total], index) => total === info.totalBytes && (!index || done >= progress[index - 1][0])));
  assert.deepEqual(new Uint8Array(await (await (await directory.getFileHandle('b.onnx')).getFile()).arrayBuffer()), bytes);
  assert.ok(await directory.getDirectoryHandle('references'));
  allRequests.length = 0;
  requests.length = 0;
  await downloadLocalBundle(models, manifest, 'B', directory, () => {}, combinedFetch);
  assert.deepEqual(allRequests, []);
  await assert.rejects(downloadLocalBundle(models, { ...manifest, version: 99 }, 'B', new MemoryDirectory('invalid-pack'), () => {}, combinedFetch), /Invalid reference manifest/);
  assert.deepEqual(allRequests, []);
});

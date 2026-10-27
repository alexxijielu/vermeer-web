import test from 'node:test';
import assert from 'node:assert/strict';
import { lookupUniProt, uniprotURL, variantFiles } from '../web/core.js';
import { ModelStore } from '../web/model-store.js';

class MemoryCache {
  entries = new Map();
  async match(key) { return this.entries.get(key)?.clone(); }
  async put(key, response) { this.entries.set(key, response.clone()); }
}

test('setup downloads and verifies weights; prediction reads never access the network', async () => {
  const data = Uint8Array.from([10, 20, 30]);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data));
  const hash = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
  const manifest = { files: { 'model.bin': { bytes: 3, sha256: hash } } };
  let requests = 0;
  const cache = new MemoryCache();
  const store = new ModelStore(manifest, cache, async function () {
    assert.equal(this, undefined);
    requests++;
    return new Response(data);
  });
  await assert.rejects(store.read('model.bin'), /not prepared/);
  await store.prepare('model.bin');
  assert.equal(requests, 1);
  assert.deepEqual(await store.read('model.bin'), data);
  assert.deepEqual(await store.read('model.bin'), data);
  assert.equal(requests, 1);
  const nextSetup = new ModelStore(manifest, cache, async () => { throw new Error('No download expected'); });
  await nextSetup.prepare('model.bin');
  cache.entries.clear();
  await assert.rejects(nextSetup.read('model.bin'), /evicted/);
  assert.equal(requests, 1);
});
test('corrupt files do not become prepared weights', async () => {
  const store = new ModelStore({ files: { model: { bytes: 3, sha256: 'wrong' } } }, new MemoryCache(), async () => new Response(Uint8Array.from([1, 2, 3])));
  await assert.rejects(store.prepare('model'), /integrity/);
  await assert.rejects(store.read('model'), /not prepared/);
});
test('model setup includes every common model and only the selected variant', () => {
  const manifest = { commonModels: ['esm.onnx', 'decoder.onnx'], variants: {
    B: { graph: 'b.onnx', externalData: ['b.bin'] },
    L: { graph: 'l.onnx', externalData: ['l.bin'] },
    XL: { graph: 'xl.onnx', externalData: ['xl.bin'] },
  } };
  for (const id of ['B', 'L', 'XL']) {
    assert.deepEqual(variantFiles(manifest, id), ['esm.onnx', 'decoder.onnx', `${id.toLowerCase()}.onnx`, `${id.toLowerCase()}.bin`]);
  }
  assert.throws(() => variantFiles(manifest, 'invalid'));
});
test('UniProt lookup sends only an identifier and returns a validated FASTA record', async () => {
  let requested;
  const fasta = '>sp|P04637|P53_HUMAN\nMEEPQSDPSV';
  assert.equal(await lookupUniProt(' p04637 ', async url => { requested = url; return new Response(fasta); }), fasta);
  assert.equal(requested, 'https://rest.uniprot.org/uniprotkb/P04637.fasta');
  assert.equal(uniprotURL('P53_HUMAN'), 'https://rest.uniprot.org/uniprotkb/P53_HUMAN.fasta');
  assert.throws(() => uniprotURL('https://example.com'));
  await assert.rejects(lookupUniProt('UNKNOWN', async () => new Response('', { status: 404 })), /No UniProt entry/);
  await assert.rejects(lookupUniProt('P04637', async () => new Response('Not FASTA')), /unexpected response/);
});

test('UniProt distinguishes a retrieved but overlength protein from a failed lookup', async () => {
  const fasta = '>sp|Q8WZ42|TITIN_HUMAN\n' + 'M'.repeat(34350);
  await assert.rejects(lookupUniProt('TITIN_HUMAN', async () => new Response(fasta)), error => {
    assert.match(error.message, /UniProt entry found/);
    assert.match(error.message, /34,350 residues/);
    assert.match(error.message, /This web application accepts 3–1,022 residues/);
    assert.match(error.message, /will not be truncated/);
    return true;
  });
  await assert.rejects(lookupUniProt('TITIN_HUMAN', async () => new Response('', { status: 503 })), /UniProt lookup failed \(HTTP 503\)/);
});

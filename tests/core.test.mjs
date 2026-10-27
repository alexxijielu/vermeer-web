import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLE_SEQUENCE, parseSequence, randomGenerator, sampleToken, normalizePixels } from '../web/core.js';

test('parses raw and single-record FASTA without altering residues', () => {
  assert.equal(parseSequence('>protein\nacde fg\nHIKX').sequence, 'ACDEFGHIKX');
  assert.equal(parseSequence(EXAMPLE_SEQUENCE).sequence, EXAMPLE_SEQUENCE);
});
test('rejects unsupported characters, multiple records and out-of-range lengths', () => {
  for (const input of ['', 'MA', 'M*AA', 'MA-AA', '>a\nMAAA\n>b\nMVVV', 'M'.repeat(1023)]) {
    assert.throws(() => parseSequence(input));
  }
  assert.equal(parseSequence('M'.repeat(1022)).sequence.length, 1022);
});
test('sampling is repeatable and observes top-k', () => {
  const a = randomGenerator(42);
  const b = randomGenerator(42);
  const logits = Float32Array.from([0, 1, 2, 3]);
  for (let i = 0; i < 50; i++) assert.equal(sampleToken(logits, 1, a), sampleToken(logits, 1, b));
  assert.equal(sampleToken(logits, 1, () => 0.99, 1), 3);
  assert.throws(() => sampleToken([NaN], 1, a));
});
test('normalizes only finite 256 by 256 images and handles constant output', () => {
  const values = new Float32Array(65536).fill(5);
  assert.equal(normalizePixels(values)[0], 0);
  values[0] = 0;
  values[1] = 10;
  const normalized = normalizePixels(values);
  assert.equal(normalized[0], 0);
  assert.equal(normalized[1], 255);
  assert.equal(normalized[2], 128);
  values[3] = Infinity;
  assert.throws(() => normalizePixels(values));
  assert.throws(() => normalizePixels([0, 1]));
});

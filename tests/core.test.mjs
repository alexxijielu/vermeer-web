import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLE_SEQUENCE, parseSequence, randomGenerator, sampleToken, sampleProteinToken, normalizePixels } from '../web/core.js';

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
test('protein sampling enforces the full channel layout before top-k without clamping', () => {
  const logits = new Float32Array(16393).fill(-100);
  logits[42] = 2;
  logits[5] = 1;
  logits.fill(1000, 16384);
  const original = logits.slice();
  let draws = 0;
  const generated = Array.from({ length: 259 }, (_, step) =>
    sampleProteinToken(logits, step, 1, () => { draws++; return 0.99; }, 1));
  assert.equal(generated[0], 16387);
  assert.deepEqual(generated.slice(1, 257), new Array(256).fill(42));
  assert.equal(generated[257], 16391);
  assert.equal(generated[258], 16392);
  assert.equal(draws, 259);
  assert.deepEqual(logits, original);
});
test('constrained image sampling preserves temperature, top-k and seeded draws over allowed tokens', () => {
  const logits = new Float32Array(16393).fill(-20);
  logits[0] = 2;
  logits[17] = 1;
  logits[16383] = 0;
  logits.fill(1000, 16384);
  for (const temperature of [0.1, 1, 2]) {
    for (const topK of [1, 2, 2000]) {
      const actual = randomGenerator(42);
      const expected = randomGenerator(42);
      for (const step of [0, 1, 128, 256, 257, 258]) {
        const token = sampleProteinToken(logits, step, temperature, actual, topK);
        const boundary = { 0: 16387, 257: 16391, 258: 16392 }[step];
        if (boundary !== undefined) {
          expected();
          assert.equal(token, boundary);
        } else {
          assert.equal(token, sampleToken(logits.slice(0, 16384), temperature, expected, topK));
        }
      }
      assert.equal(actual(), expected());
    }
  }
  assert.equal(sampleProteinToken(logits, 1, 1, () => 0), 0);
  assert.equal(sampleProteinToken(logits, 1, 1, () => 0.99), 16383);
});
test('constrained sampling rejects invalid positions, vocabulary shapes and non-finite model output', () => {
  const logits = new Float32Array(16393);
  for (const step of [-1, 259, 1.5, undefined]) {
    assert.throws(() => sampleProteinToken(logits, step, 1, () => 0.5), /position/);
  }
  for (const length of [0, 16384, 16392, 16394]) {
    assert.throws(() => sampleProteinToken(new Float32Array(length), 1, 1, () => 0.5), /vocabulary/);
  }
  for (const index of [42, 16384, 16387, 16391, 16392]) {
    for (const value of [NaN, Infinity, -Infinity]) {
      logits[index] = value;
      for (const step of [0, 1, 257, 258]) {
        assert.throws(() => sampleProteinToken(logits, step, 1, () => 0.5), /non-finite/);
      }
      logits[index] = 0;
    }
  }
  for (const step of [0, 1, 257, 258]) {
    assert.throws(() => sampleProteinToken(logits, step, 0, () => 0.5), /sampling settings/);
    assert.throws(() => sampleProteinToken(logits, step, 1, () => 0.5, 0), /sampling settings/);
  }
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

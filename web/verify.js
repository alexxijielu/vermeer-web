import * as ort from './runtime/ort.webgpu.min.mjs';
import { VermeerInference } from './inference.js';

function compare(name, actual, expected, atol) {
  if (actual.length !== expected.length) throw new Error(`${name}: shape mismatch.`);
  let maxError = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(error) || error > atol + 0.001 * Math.abs(expected[i])) {
      throw new Error(`${name}: numerical mismatch at ${i}: ${actual[i]} vs ${expected[i]}.`);
    }
    maxError = Math.max(maxError, error);
  }
  return maxError;
}

export async function verifyBrowser(variant = 'B', directory) {
  if (!directory) throw new Error('Pass the selected local data folder handle to verifyBrowser.');
  const response = await fetch(`./assets/browser-fixture-${variant.toLowerCase()}.json`);
  if (!response.ok) throw new Error('Run tools\\make_browser_fixture.py first.');
  const fixture = await response.json();
  const engine = new VermeerInference(message => console.info(message));
  await engine.setup(variant, () => {}, directory);
  const embedding = await engine.embed(fixture.sequence);
  const report = { device: engine.deviceInfo, embeddingMaxError: compare('ESM-C', embedding, fixture.embedding, 0.002), cachedGPTMaxErrors: [] };
  await engine.loadGPT();
  const conditioning = new ort.Tensor('float32', embedding, [1, 1, 1152]);
  let past = engine.makePast();
  try {
    for (let step = 0; step < 4; step++) {
      const ids = step === 0 ? [0, ...engine.reference.prefix] : [fixture.gptTokens[step - 1]];
      const positions = step === 0 ? Array.from({ length: 775 }, (_, i) => i) : [774 + step];
      const tokens = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const positionTensor = new ort.Tensor('int64', BigInt64Array.from(positions, BigInt), [positions.length]);
      const feeds = { tokens, conditioning, positions: positionTensor };
      past.forEach((tensor, index) => { feeds[`past_${index}`] = tensor; });
      const output = await engine.gpt.run(feeds);
      past.forEach(tensor => tensor.dispose());
      past = past.map((_, index) => output[`present_${index}`]);
      tokens.dispose();
      positionTensor.dispose();
      report.cachedGPTMaxErrors.push(compare(`GPT step ${step}`, output.logits.data, fixture.gptLogits[step], 0.003));
      output.logits.dispose();
    }
  } finally {
    past.forEach(tensor => tensor.dispose());
    conditioning.dispose();
    await engine.gpt.release();
    engine.gpt = null;
  }
  const decoder = await engine.loadModel('decoder.onnx');
  const tokens = new ort.Tensor('int64', BigInt64Array.from(fixture.decoderTokens, BigInt), [1, 256]);
  try {
    const output = await decoder.run({ tokens });
    report.decoderMaxError = compare('Decoder', output.image.data, fixture.decoderPixels, 0.003);
    output.image.dispose();
  } finally {
    tokens.dispose();
    await decoder.release();
  }
  return report;
}

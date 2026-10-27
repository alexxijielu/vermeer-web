import * as ort from './runtime/ort.webgpu.min.mjs';
import { randomGenerator, sampleToken, variantFiles } from './core.js';
import { FolderModelStore } from './model-store.js';

ort.env.wasm.wasmPaths = new URL('./runtime/', import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

const int64 = (values, dims) => new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Missing ${url} (HTTP ${response.status}). Run the model export before setting up inference.`);
  return response.json();
}

export class VermeerInference {
  constructor(update) {
    this.update = update;
    this.gpt = null;
    this.embeddingCache = new Map();
    this.cancelled = false;
  }

  checkCancellation() {
    if (this.cancelled) throw new DOMException('Prediction cancelled.', 'AbortError');
  }

  async setup(variant = 'B', onProgress = () => {}, directory = null) {
    this.cancelled = false;
    if (!directory) throw new Error('Choose a local data folder or download the local files before setup.');
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Try an up-to-date Chrome or Edge browser with hardware acceleration enabled.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available. Enable hardware acceleration or use a supported GPU.');
    const info = adapter.info;
    this.deviceInfo = `${info.vendor || 'GPU'} ${info.architecture || ''} ${info.device || ''}`.trim();
    this.manifest = await json('./assets/manifest.json');
    if (this.manifest.version !== 2) throw new Error('Re-export the browser models for the current multi-model application.');
    this.variant = variant;
    this.model = this.manifest.variants[variant];
    const files = variantFiles(this.manifest, variant);
    this.reference = await json('./assets/reference.json');
    this.vocab = await json('./assets/tokenizer.json');
    this.store = new FolderModelStore(this.manifest, directory);
    const total = files.reduce((sum, name) => sum + this.manifest.files[name].bytes, 0);
    let completed = 0;
    for (const [index, name] of files.entries()) {
      this.update(`Preparing all model files: ${index + 1} / ${files.length} (${(completed / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB)…`);
      await this.store.prepare(name);
      completed += this.manifest.files[name].bytes;
      onProgress(completed, total);
    }
    for (const [index, name] of this.manifest.commonModels.entries()) {
      this.update(`Checking local model ${index + 1} / ${this.manifest.commonModels.length}: ${name}…`);
      const session = await this.loadModel(name);
      await session.release();
    }
    await this.loadGPT();
    return { device: this.deviceInfo, manifest: this.manifest, model: this.model, totalBytes: total };
  }

  async loadModel(name, locations = 'cpu') {
    this.checkCancellation();
    const bytes = await this.store.read(name);
    const externalData = [];
    if (name === this.model.graph) {
      for (const path of this.model.externalData) {
        externalData.push({ path, data: await this.store.read(path) });
      }
    }
    this.checkCancellation();
    return ort.InferenceSession.create(bytes, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: locations,
      graphOptimizationLevel: 'all',
      externalData,
    });
  }

  async loadGPT() {
    if (!this.gpt) {
      const locations = { logits: 'cpu' };
      for (let index = 0; index < 2 * this.model.layers; index++) locations[`present_${index}`] = 'gpu-buffer';
      this.gpt = await this.loadModel(this.model.graph, locations);
    }
  }

  makePast() {
    const shape = [1, this.model.heads, 1, this.model.headDim];
    return Array.from({ length: 2 * this.model.layers }, () =>
      new ort.Tensor('float32', new Float32Array(this.model.heads * this.model.headDim), shape));
  }

  async dispose() {
    if (this.gpt) await this.gpt.release();
    this.gpt = null;
  }

  async embed(sequence) {
    const cached = this.embeddingCache.get(sequence);
    if (cached) {
      this.update('Reusing the sequence embedding from this tab.');
      return cached;
    }
    if (this.gpt) {
      await this.gpt.release();
      this.gpt = null;
    }
    const ids = [0, ...Array.from(sequence, letter => {
      const value = this.vocab[letter];
      if (value === undefined) throw new Error(`The ESM-C tokenizer does not support ${letter}.`);
      return value;
    }), 2];
    let hidden;
    let session = await this.loadModel('esm-start.onnx');
    try {
      const output = await session.run({ tokens: int64(ids, [1, ids.length]) });
      hidden = output.hidden;
    } finally {
      await session.release();
    }
    const positions = int64(Array.from({ length: ids.length }, (_, index) => index), [ids.length]);
    try {
      for (let index = 0; index < 36; index++) {
        this.checkCancellation();
        this.update(`Embedding protein: ESM-C layer ${index + 1} of 36…`);
        session = await this.loadModel(`esm-layer-${String(index).padStart(2, '0')}.onnx`);
        try {
          const output = await session.run({ hidden, positions });
          hidden.dispose();
          hidden = output.output;
        } finally {
          await session.release();
        }
      }
      this.checkCancellation();
      session = await this.loadModel('esm-end.onnx');
      try {
        const output = await session.run({ hidden });
        const embedding = new Float32Array(output.embedding.data);
        output.embedding.dispose();
        if (embedding.length !== 1152 || embedding.some(value => !Number.isFinite(value))) {
          throw new Error('ESM-C returned an invalid embedding.');
        }
        if (this.embeddingCache.size >= 4) this.embeddingCache.delete(this.embeddingCache.keys().next().value);
        this.embeddingCache.set(sequence, embedding);
        return embedding;
      } finally {
        await session.release();
      }
    } finally {
      hidden?.dispose();
      positions.dispose();
    }
  }

  async predict(sequence, { seed, temperature }, onProgress) {
    this.cancelled = false;
    const started = performance.now();
    const embedding = await this.embed(sequence);
    this.checkCancellation();
    await this.loadGPT();
    const random = randomGenerator(seed);
    const conditioning = new ort.Tensor('float32', embedding, [1, 1, 1152]);
    let past = this.makePast();
    let ids = [0, ...this.reference.prefix];
    let positions = Array.from({ length: ids.length }, (_, index) => index);
    const generated = [];
    try {
      for (let step = 0; step < 259; step++) {
        this.checkCancellation();
        this.update(step === 0 ? 'Reading the reference stains…' : `Generating protein channel: ${step} / 259 tokens…`);
        const tokens = int64(ids, [1, ids.length]);
        const positionTensor = int64(positions, [positions.length]);
        let output;
        try {
          const feeds = { tokens, conditioning, positions: positionTensor };
          past.forEach((tensor, index) => { feeds[`past_${index}`] = tensor; });
          output = await this.gpt.run(feeds);
        } finally {
          tokens.dispose();
          positionTensor.dispose();
        }
        past.forEach(tensor => tensor.dispose());
        past = past.map((_, index) => output[`present_${index}`]);
        let token;
        try {
          token = sampleToken(output.logits.data, temperature, random);
        } finally {
          output.logits.dispose();
        }
        generated.push(token);
        ids = [token];
        positions = [775 + step];
        onProgress(step + 1);
      }
    } finally {
      past.forEach(tensor => tensor.dispose());
      conditioning.dispose();
      await this.gpt.release();
      this.gpt = null;
    }
    this.checkCancellation();
    const imageCodes = generated.slice(1, 257);
    if (imageCodes.length !== 256 || imageCodes.some(token => token < 0 || token >= 16384)) {
      throw new Error('The model sampled a channel-boundary token inside the image. No image was fabricated; try a different seed or lower temperature.');
    }
    const expected = [16387, 16391, 16392];
    if (generated[0] !== expected[0] || generated[257] !== expected[1] || generated[258] !== expected[2]) {
      throw new Error('The generated channel has invalid start/end tokens. Try a different seed or lower temperature.');
    }
    this.update('Decoding the predicted protein image…');
    const decoder = await this.loadModel('decoder.onnx');
    const tokens = int64(imageCodes, [1, 256]);
    try {
      const output = await decoder.run({ tokens });
      try {
        this.checkCancellation();
        return {
          pixels: new Float32Array(output.image.data),
          tokens: generated, embedding, seconds: (performance.now() - started) / 1000,
          device: this.deviceInfo, model: this.model.name, precision: this.manifest.precision,
        };
      } finally {
        output.image.dispose();
      }
    } finally {
      tokens.dispose();
      await decoder.release();
    }
  }
}

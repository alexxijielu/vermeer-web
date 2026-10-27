import { ModelStore, FolderModelStore, downloadFolderFiles } from './model-store.js';

const options = { basePath: './assets/', label: 'reference' };
const imageNames = ['reference.png', 'landmark-nucleus.png', 'landmark-microtubules.png', 'landmark-er.png'];

export function referenceFiles(manifest) {
  if (manifest.version !== 1 || !Array.isArray(manifest.browserFiles) || !manifest.browserFiles.includes('references.json')) {
    throw new Error('Invalid reference manifest. Prepare the reference collection again.');
  }
  const files = manifest.browserFiles;
  if (new Set(files).size !== files.length || files.length !== Object.keys(manifest.files).length) {
    throw new Error('The reference manifest contains duplicate or unexpected files.');
  }
  for (const name of files) {
    const entry = manifest.files[name];
    if (!/^(?:references\.json|references\/[A-Za-z0-9_-]+\/\d{2}(?:\.json|\/(?:reference|landmark-nucleus|landmark-microtubules|landmark-er)\.png))$/.test(name) ||
        !entry || !Number.isSafeInteger(entry.bytes) || entry.bytes <= 0 || !/^[a-f0-9]{64}$/.test(entry.sha256)) {
      throw new Error(`Invalid reference manifest entry: ${name}. Only prepared PNGs and JSON tokens are supported.`);
    }
  }
  return [...files];
}

export class ReferenceStore {
  constructor(manifest, directory = null, fetcher = fetch) {
    referenceFiles(manifest);
    this.manifest = manifest;
    this.local = directory ? new FolderModelStore(manifest, directory, options) : null;
    this.verifier = this.local || new ModelStore(manifest, null, fetcher, options);
    this.fetcher = (...args) => fetcher(...args);
  }

  async read(name, signal) {
    signal?.throwIfAborted();
    const url = this.verifier.key(name);
    let bytes;
    if (this.local) {
      bytes = await this.local.bytes(name);
    } else {
      const response = await this.fetcher(url, { signal, cache: 'no-store' });
      if (!response.ok) throw new Error(`Could not load reference file ${name}: HTTP ${response.status}.`);
      bytes = new Uint8Array(await response.arrayBuffer());
    }
    signal?.throwIfAborted();
    await this.verifier.verify(name, bytes);
    signal?.throwIfAborted();
    return bytes;
  }

  async json(name, signal) {
    const bytes = await this.read(name, signal);
    try {
      return JSON.parse(new TextDecoder().decode(bytes));
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(`Invalid reference JSON: ${name}.`, { cause: error });
      throw error;
    }
  }

  async catalog(signal) {
    const catalog = await this.json('references.json', signal);
    const lines = catalog.cellLines;
    if (catalog.version !== 2 || catalog.fieldsPerCellLine !== 10 || !lines || Object.keys(lines).length !== this.manifest.cellLineCount) {
      throw new Error('Invalid reference catalog or incompatible collection version.');
    }
    for (const [name, fields] of Object.entries(lines)) {
      if (!Array.isArray(fields) || fields.length !== catalog.fieldsPerCellLine ||
          fields.some((entry, index) => entry.index !== index + 1 || typeof entry.id !== 'string' ||
            typeof entry.url !== 'string' || !entry.url.startsWith('/assets/references/') ||
            !this.manifest.files[entry.url.slice('/assets/'.length)])) {
        throw new Error(`Invalid reference fields for ${name}.`);
      }
    }
    return catalog;
  }

  async load(entry, cellLine, signal) {
    const path = entry.url.slice('/assets/'.length);
    const reference = await this.json(path, signal);
    const expectedImages = imageNames.map(name => `${path.slice(0, -5)}/${name}`);
    if (reference.id !== entry.id || reference.cellLine !== cellLine || reference.index !== entry.index ||
        !Array.isArray(reference.images) || reference.images.length !== 4 ||
        reference.images.some((name, index) => name !== expectedImages[index]) ||
        !Array.isArray(reference.prefix) || reference.prefix.length !== 774) {
      throw new Error(`Invalid reference field: ${cellLine} / ${entry.index}.`);
    }
    for (let channel = 0; channel < 3; channel++) {
      const tokens = reference.prefix.slice(channel * 258, (channel + 1) * 258);
      if (tokens[0] !== 16384 + channel || tokens.at(-1) !== 16388 + channel ||
          tokens.slice(1, -1).some(token => !Number.isInteger(token) || token < 0 || token >= 16384)) {
        throw new Error(`Invalid landmark tokens: ${cellLine} / ${entry.index}.`);
      }
    }
    const images = new Map(await Promise.all(reference.images.map(async name =>
      [name.split('/').at(-1), await this.read(name, signal)])));
    return { reference, images };
  }
}

export async function downloadReferenceFolder(manifest, directory, onProgress = () => {}, fetcher = fetch) {
  return downloadFolderFiles(manifest, referenceFiles(manifest), directory, onProgress, fetcher, options);
}

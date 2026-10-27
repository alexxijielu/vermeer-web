import { variantFiles } from './core.js';

class ModelIntegrityError extends Error {}

export class ModelStore {
  constructor(manifest, cache, fetcher = fetch, { basePath = manifest.downloadBaseUrl ?? './assets/models/', label = 'model' } = {}) {
    this.manifest = manifest;
    this.cache = cache;
    this.fetcher = (...args) => fetcher(...args);
    this.prepared = new Set();
    this.basePath = basePath;
    this.label = label;
  }

  key(name) {
    const entry = this.manifest.files[name];
    if (!entry) throw new Error(`The ${this.label} manifest does not include ${name}.`);
    return new URL(`${this.basePath}${name}?sha256=${entry.sha256}`, globalThis.location?.href || 'http://localhost/').href;
  }

  async verify(name, bytes) {
    const entry = this.manifest.files[name];
    if (bytes.byteLength !== entry.bytes) throw new ModelIntegrityError(`Incomplete ${this.label} file: ${name}. Download or update the ${this.label} files, then try again.`);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== entry.sha256) throw new ModelIntegrityError(`${this.label} integrity check failed: ${name}. Download or update the ${this.label} files.`);
  }

  async prepare(name) {
    const key = this.key(name);
    let response = await this.cache.match(key);
    if (!response) {
      response = await this.fetcher(key);
      if (!response.ok) throw new Error(`Could not download ${name}: HTTP ${response.status}.`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      await this.verify(name, bytes);
      await this.cache.put(key, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } else {
      await this.verify(name, new Uint8Array(await response.arrayBuffer()));
    }
    this.prepared.add(name);
  }

  async read(name) {
    if (!this.prepared.has(name)) throw new Error(`${name} was not prepared. Run model setup first.`);
    const response = await this.cache.match(this.key(name));
    if (!response) throw new Error(`The browser evicted ${name} from local storage. Run model setup again; prediction will not download missing models.`);
    return new Uint8Array(await response.arrayBuffer());
  }
}

export class FolderModelStore extends ModelStore {
  constructor(manifest, directory, options = {}) {
    super(manifest, null, fetch, options);
    this.directory = directory;
  }

  async handle(name, create = false) {
    this.key(name);
    const parts = name.split('/');
    if (parts.some(part => !part || part === '.' || part === '..' || part.includes('\\'))) {
      throw new Error(`Invalid model path: ${name}.`);
    }
    let directory = this.directory;
    for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part, { create });
    return directory.getFileHandle(parts.at(-1), { create });
  }

  async bytes(name) {
    try {
      const file = await (await this.handle(name)).getFile();
      if (file.size !== this.manifest.files[name].bytes) {
        throw new ModelIntegrityError(`Incomplete ${this.label} file: ${name}. Download or update the ${this.label} files, then try again.`);
      }
      return new Uint8Array(await file.arrayBuffer());
    } catch (error) {
      if (error.name === 'NotFoundError') throw new Error(`Missing local ${this.label} file: ${name}. Choose the correct folder or use Download / update local files.`, { cause: error });
      if (error.name === 'NotAllowedError') throw new Error(`Permission to read the ${this.label} folder was lost. Choose the folder again.`, { cause: error });
      throw error;
    }
  }

  async prepare(name) {
    await this.verify(name, await this.bytes(name));
    this.prepared.add(name);
  }

  async read(name) {
    if (!this.prepared.has(name)) throw new Error(`${name} was not prepared. Run model setup first.`);
    const bytes = await this.bytes(name);
    await this.verify(name, bytes);
    return bytes;
  }
}

export async function downloadModelFolder(manifest, variant, directory, onProgress = () => {}, fetcher = fetch) {
  return downloadFolderFiles(manifest, variantFiles(manifest, variant), directory, onProgress, fetcher);
}

export async function downloadFolderFiles(manifest, files, directory, onProgress = () => {}, fetcher = fetch, options = {}) {
  const store = new FolderModelStore(manifest, directory, options);
  const total = files.reduce((sum, name) => sum + manifest.files[name].bytes, 0);
  let completed = 0;
  for (const name of files) {
    onProgress(completed, total, `Checking ${name}…`);
    let valid = false;
    try {
      await store.verify(name, await store.bytes(name));
      valid = true;
    } catch (error) {
      if (error.cause?.name !== 'NotFoundError' && !(error instanceof ModelIntegrityError)) throw error;
    }
    if (!valid) {
      onProgress(completed, total, `Downloading ${name}…`);
      const response = await fetcher(store.key(name), { cache: 'no-store' });
      if (!response.ok) throw new Error(`Could not download ${name}: HTTP ${response.status}. Run Download / update again to resume.`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      await store.verify(name, bytes);
      const writable = await (await store.handle(name, true)).createWritable();
      try {
        await writable.write(bytes);
        await writable.close();
      } catch (error) {
        try {
          await writable.abort();
        } catch (cleanupError) {
          throw new AggregateError([error, cleanupError], `Could not save ${name}: ${error.message}. Temporary-write cleanup also failed: ${cleanupError.message}.`);
        }
        throw error;
      }
    }
    completed += manifest.files[name].bytes;
    onProgress(completed, total, `Saved / verified ${name}.`);
  }
  return { totalBytes: total, files: files.length };
}

export class MemoryDirectory {
  directories = new Map();
  files = new Map();
  async getDirectoryHandle(name, { create = false } = {}) {
    if (!this.directories.has(name)) {
      if (!create) throw new DOMException('Missing directory', 'NotFoundError');
      this.directories.set(name, new MemoryDirectory());
    }
    return this.directories.get(name);
  }
  async getFileHandle(name, { create = false } = {}) {
    if (!this.files.has(name)) {
      if (!create) throw new DOMException('Missing file', 'NotFoundError');
      this.files.set(name, new Uint8Array());
    }
    return {
      getFile: async () => new Blob([this.files.get(name)]),
      createWritable: async () => {
        let pending;
        return {
          write: async bytes => { pending = bytes.slice(); },
          close: async () => { this.files.set(name, pending); },
          abort: async () => {},
        };
      },
    };
  }
}

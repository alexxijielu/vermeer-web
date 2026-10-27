import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = path.join(root, 'web');
const output = path.join(root, 'dist');
const config = JSON.parse(await readFile(path.join(root, 'deployment.json'), 'utf8'));
if (!/^[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+$/.test(config.modelRepository) || !/^[a-f0-9]{40}$/.test(config.modelRevision)) {
  throw new Error('deployment.json must identify a public model repository and a verified, pinned 40-character commit revision.');
}

await rm(output, { recursive: true, force: true });
await cp(source, output, {
  recursive: true,
  filter(filename) {
    const relative = path.relative(source, filename);
    const parts = relative.split(path.sep);
    return !parts.some(part => part.startsWith('.')) &&
      !(parts[0] === 'assets' && parts[1] === 'models') &&
      !/^browser-fixture(?:-[bl]|-xl)?\.json$/.test(path.basename(filename));
  },
});
const runtime = path.join(output, 'runtime');
await mkdir(runtime);
for (const name of ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  await cp(path.join(root, 'node_modules', 'onnxruntime-web', 'dist', name), path.join(runtime, name));
}
await mkdir(path.join(output, 'vendor'));
await cp(path.join(root, 'node_modules', 'fflate', 'esm', 'browser.js'), path.join(output, 'vendor', 'fflate.js'));
await cp(path.join(root, 'licenses'), path.join(output, 'licenses'), { recursive: true });
await cp(path.join(root, 'node_modules', 'fflate', 'LICENSE'), path.join(output, 'licenses', 'fflate.txt'));

const manifestPath = path.join(output, 'assets', 'manifest.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
manifest.downloadBaseUrl = `https://huggingface.co/${config.modelRepository}/resolve/${config.modelRevision}/`;
manifest.notices = [];
for (const name of (await readdir(path.join(root, 'licenses', 'models'))).sort()) {
  const bytes = await readFile(path.join(root, 'licenses', 'models', name));
  const filename = `licenses/${name}`;
  manifest.notices.push(filename);
  manifest.files[filename] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(path.join(output, '.nojekyll'), '');
console.log(`Built static site in dist; model downloads pinned to ${config.modelRepository}@${config.modelRevision}.`);

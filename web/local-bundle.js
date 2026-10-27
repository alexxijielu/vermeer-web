import { variantFiles } from './core.js';
import { downloadModelFolder } from './model-store.js';
import { downloadReferenceFolder, referenceFiles } from './reference-store.js';

export async function downloadLocalBundle(modelManifest, referenceManifest, variant, directory, onProgress = () => {}, fetcher = fetch) {
  const models = variantFiles(modelManifest, variant);
  const references = referenceFiles(referenceManifest);
  if (models.some(name => references.includes(name))) throw new Error('Model and reference bundle paths overlap.');
  const modelBytes = models.reduce((sum, name) => sum + modelManifest.files[name].bytes, 0);
  const referenceBytes = references.reduce((sum, name) => sum + referenceManifest.files[name].bytes, 0);
  const total = modelBytes + referenceBytes;
  const modelResult = await downloadModelFolder(modelManifest, variant, directory,
    (done, _, message) => onProgress(done, total, `Models: ${message}`), fetcher);
  const referenceResult = await downloadReferenceFolder(referenceManifest, directory,
    (done, _, message) => onProgress(modelBytes + done, total, `References: ${message}`), fetcher);
  return { totalBytes: total, files: modelResult.files + referenceResult.files };
}

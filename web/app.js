import { EXAMPLE_SEQUENCE, parseSequence, normalizePixels, lookupUniProt } from './core.js';
import { ReferenceStore } from './reference-store.js';
import { downloadLocalBundle } from './local-bundle.js';

const $ = id => document.getElementById(id);
let engine;
let ready = false;
let busy = false;
let result;
let lookupBusy = false;
let referenceLoading = true;
let referenceCatalog;
let selectedReference;
let referenceImages = new Map();
let referenceVersion = 0;
let referenceAbort;
let referenceStore;
let referenceImageURL = null;
let modelDirectory = null;
const folderSupported = typeof globalThis.showDirectoryPicker === 'function';
const showError = error => {
  console.error(error);
  $('error').textContent = error.message || String(error);
  $('error').hidden = false;
};
function updateControls() {
  $('setup').disabled = busy || ready || !modelDirectory || referenceLoading || !selectedReference;
  $('choose-models').disabled = busy || !folderSupported;
  $('download-models').disabled = busy || !folderSupported;
  $('generate').disabled = !ready || busy || lookupBusy || referenceLoading || !selectedReference;
  $('model-variant').disabled = busy;
  for (const id of ['sequence', 'fasta', 'seed', 'temperature', 'example', 'uniprot-id', 'uniprot-lookup']) $(id).disabled = busy || lookupBusy;
  $('cell-line').disabled = busy || !referenceCatalog;
  $('reference-field').disabled = busy || !referenceCatalog;
}
function clearResult() {
  result = null;
  $('prediction').hidden = true;
  $('prediction-placeholder').hidden = false;
  $('downloads').hidden = true;
  $('result-meta').textContent = '';
  $('prediction-caption').textContent = 'Green: model-generated protein signal';
  $('results-description')?.replaceChildren('A reference image and predicted protein channel will appear here after inference.');
}
async function getJSON(url, signal) {
  const response = await fetch(url, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}. Prepare the reference images first.`);
  return response.json();
}
function startReferenceLoad() {
  referenceAbort?.abort();
  referenceAbort = new AbortController();
  const version = ++referenceVersion;
  referenceLoading = true;
  selectedReference = null;
  referenceImages = new Map();
  if (engine) engine.reference = null;
  if (referenceImageURL) URL.revokeObjectURL(referenceImageURL);
  referenceImageURL = null;
  clearResult();
  $('reference-preview').hidden = true;
  $('reference').hidden = true;
  $('reference-preview').removeAttribute('src');
  $('reference').removeAttribute('src');
  $('reference-placeholder').hidden = false;
  $('preview-placeholder').hidden = false;
  $('preview-placeholder').textContent = 'Loading selected landmarks…';
  $('reference-description').textContent = 'Loading selected reference…';
  $('reference-sources')?.replaceChildren('Loading TIFF sources…');
  updateControls();
  return { version, signal: referenceAbort.signal };
}
async function selectReference() {
  const { version, signal } = startReferenceLoad();
  try {
    const entry = referenceCatalog.cellLines[$('cell-line').value].find(item => String(item.index) === $('reference-field').value);
    const { reference, images } = await referenceStore.load(entry, $('cell-line').value, signal);
    if (version !== referenceVersion) return;
    selectedReference = reference;
    referenceImages = images;
    referenceImageURL = URL.createObjectURL(new Blob([images.get('reference.png')], { type: 'image/png' }));
    for (const id of ['reference-preview', 'reference']) {
      $(id).src = referenceImageURL;
      $(id).alt = `${reference.cellLine} image ${reference.index}: blue nucleus, red microtubules, yellow ER; no protein channel`;
      $(id).hidden = false;
    }
    $('reference-placeholder').hidden = true;
    $('preview-placeholder').hidden = true;
    $('reference-description').textContent = `${reference.cellLine} · image ${reference.index} of ${referenceCatalog.cellLines[reference.cellLine].length} · ${reference.antibody} / field ${reference.fov} · TIFF-derived landmarks`;
    $('reference-sources')?.replaceChildren(...Object.entries(reference.provenance).flatMap(([name, source], index) => {
      const link = document.createElement('a');
      link.href = source.url;
      link.textContent = `${name} TIFF (${source.originalDtype === 'uint16' ? '16' : '8'}-bit source)`;
      return index === 0 ? [link] : [document.createTextNode(' · '), link];
    }));
  } catch (error) {
    if (version !== referenceVersion) return;
    referenceAbort.abort();
    $('preview-placeholder').textContent = 'Landmark preview could not be loaded.';
    $('reference-description').textContent = 'No reference selected.';
    $('reference-sources')?.replaceChildren('No reference selected.');
    showError(error);
  } finally {
    if (version === referenceVersion) {
      referenceLoading = false;
      updateControls();
    }
  }
}
function populateFields() {
  const entries = referenceCatalog.cellLines[$('cell-line').value];
  $('reference-field').replaceChildren(...entries.map(item => new Option(`Image ${item.index}`, String(item.index))));
}
async function initializeReferences() {
  const previousLine = $('cell-line').value;
  const previousField = $('reference-field').value;
  referenceCatalog = null;
  const { version, signal } = startReferenceLoad();
  if (!modelDirectory) {
    referenceLoading = false;
    $('preview-placeholder').textContent = 'Choose a local data folder first';
    $('reference-description').textContent = 'No local reference folder selected.';
    $('reference-sources')?.replaceChildren('Choose a local data folder to see source TIFF links.');
    updateControls();
    return false;
  }
  try {
    const manifest = await getJSON('./assets/reference-manifest.json', signal);
    const store = new ReferenceStore(manifest, modelDirectory);
    const catalog = await store.catalog(signal);
    if (version !== referenceVersion) return;
    referenceStore = store;
    referenceCatalog = catalog;
    $('cell-line').replaceChildren(...Object.keys(referenceCatalog.cellLines).map(name => new Option(name, name)));
    if (referenceCatalog.cellLines[previousLine]) $('cell-line').value = previousLine;
    else if (referenceCatalog.cellLines.Rh30) $('cell-line').value = 'Rh30';
    populateFields();
    if (referenceCatalog.cellLines[$('cell-line').value].some(entry => String(entry.index) === previousField)) {
      $('reference-field').value = previousField;
    }
    await selectReference();
    return Boolean(selectedReference);
  } catch (error) {
    if (version !== referenceVersion) return;
    referenceLoading = false;
    $('preview-placeholder').textContent = 'Reference collection could not be loaded.';
    $('reference-description').textContent = 'No reference selected.';
    $('reference-sources')?.replaceChildren('No reference selected.');
    showError(error);
    updateControls();
    return false;
  }
}
$('cell-line').addEventListener('change', () => { populateFields(); void selectReference(); });
$('reference-field').addEventListener('change', () => { void selectReference(); });
async function resetSetup(message) {
  busy = true;
  ready = false;
  clearResult();
  $('setup-status').classList.remove('ready');
  $('setup-status').textContent = message;
  $('run-status').textContent = 'Set up the selected model before making a prediction.';
  updateControls();
  try { await engine?.dispose(); } catch (error) { showError(error); }
  finally { busy = false; updateControls(); }
}
$('model-variant').addEventListener('change', () => {
  void resetSetup(`Vermeer-${$('model-variant').value} selected. Run setup to prepare it.`);
});
async function chooseModelFolder(download) {
  busy = true;
  let picked = false;
  $('error').hidden = true;
  updateControls();
  try {
    const directory = await globalThis.showDirectoryPicker({ mode: download ? 'readwrite' : 'read', id: 'vermeer-data' });
    picked = true;
    await resetSetup('Local data folder selected. Checking reference images…');
    busy = true;
    modelDirectory = directory;
    $('data-folder').value = directory.name;
    $('reference-folder').value = `${directory.name}\\references`;
    $('folder-status').textContent = 'Models and reference images use this one local folder. Select it again on future visits.';
    referenceCatalog = null;
    startReferenceLoad();
    updateControls();
    if (download) {
      $('setup-progress').hidden = false;
      const [manifest, references] = await Promise.all([getJSON('./assets/manifest.json'), getJSON('./assets/reference-manifest.json')]);
      await downloadLocalBundle(manifest, references, $('model-variant').value, directory, (done, total, message) => {
        $('setup-progress').max = total;
        $('setup-progress').value = done;
        $('setup-status').textContent = `${message} (${(done / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB)`;
      });
    }
    const available = await initializeReferences();
    $('setup-status').textContent = available ? `Local models and references selected in ${directory.name}. Click Set up model to verify and activate the model.` : 'Reference files are incomplete or inaccessible. Use Download / update local files.';
  } catch (error) {
    if (error.name === 'AbortError' && !picked) {
      $('folder-status').textContent = modelDirectory ? `Folder selection cancelled. Still using ${modelDirectory.name}.` : 'Folder selection cancelled. No model folder selected.';
    } else {
      referenceLoading = false;
      $('preview-placeholder').textContent = 'Local folder preparation failed';
      $('reference-description').textContent = 'Download / update local files to complete the reference collection.';
      $('setup-status').textContent = 'Local folder operation failed. See the error below; Download / update local files can resume completed files.';
      showError(error);
    }
  } finally {
    busy = false;
    $('setup-progress').hidden = true;
    updateControls();
  }
}
$('choose-models').addEventListener('click', () => { void chooseModelFolder(false); });
$('download-models').addEventListener('click', () => { void chooseModelFolder(true); });
if (!folderSupported) {
  $('folder-status').textContent = 'Folder access is unavailable here. Use a current Chrome or Edge on localhost / HTTPS.';
}
function updateCount() {
  const length = $('sequence').value.split(/\r?\n/).filter(line => !line.trim().startsWith('>')).join('').replace(/\s/g, '').length;
  $('sequence-count').textContent = `${length.toLocaleString()} residues`;
}
$('sequence').addEventListener('input', () => { updateCount(); $('uniprot-status').textContent = ''; });
$('example').addEventListener('click', () => {
  $('sequence').value = `>A0A087WSY6 | IGKV3D-15 | Homo sapiens\n${EXAMPLE_SEQUENCE}`;
  $('uniprot-status').textContent = '';
  updateCount();
});
$('uniprot-lookup').addEventListener('click', async () => {
  lookupBusy = true;
  $('error').hidden = true;
  $('uniprot-status').textContent = 'Looking up the identifier in UniProt…';
  updateControls();
  try {
    const fasta = await lookupUniProt($('uniprot-id').value);
    $('sequence').value = fasta;
    updateCount();
    $('uniprot-status').textContent = `Retrieved from UniProt: ${fasta.split(/\r?\n/)[0].slice(1)}`;
  } catch (error) {
    $('uniprot-status').textContent = `${error.message || String(error)} The existing sequence was not changed.`;
    showError(error);
  } finally {
    lookupBusy = false;
    updateControls();
  }
});
$('fasta').addEventListener('change', async event => {
  $('error').hidden = true;
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 100_000) throw new Error('Select a single FASTA record in a file smaller than 100 KB.');
    const text = await file.text();
    parseSequence(text);
    $('sequence').value = text;
    $('uniprot-status').textContent = '';
    updateCount();
  } catch (error) {
    showError(error);
  }
});
$('setup').addEventListener('click', async () => {
  busy = true;
  updateControls();
  $('error').hidden = true;
  $('setup-progress').hidden = false;
  try {
    $('setup-status').textContent = 'Checking WebGPU and loading model files…';
    const { VermeerInference } = await import('./inference.js');
    engine ??= new VermeerInference(message => {
      $(ready ? 'run-status' : 'setup-status').textContent = message;
    });
    const info = await engine.setup($('model-variant').value, (done, total) => {
      $('setup-progress').max = total;
      $('setup-progress').value = done;
    }, modelDirectory);
    const source = `Local folder: ${modelDirectory.name}; references: ${modelDirectory.name}\\references (no browser-cache copy)`;
    $('device-info').textContent = `${info.device}\n${info.model.name} · ${info.manifest.precision} · WebGPU\nESM-C-600M · 36 sequential layers\nAll ${(info.totalBytes / 1e9).toFixed(2)} GB of model files prepared locally\n${source}\nGPU stages activate sequentially; no model downloads during prediction\nPooling: legacy preprocessing compatibility (double trim)`;
    $('setup-status').textContent = `${info.model.name}, ESM-C and decoder prepared. Ready to predict.`;
    $('setup-status').classList.add('ready');
    $('run-status').textContent = 'Ready for a protein sequence.';
    ready = true;
  } catch (error) {
    $('setup-status').textContent = 'Setup failed. See the error below.';
    showError(error);
  } finally {
    busy = false;
    $('setup-progress').hidden = true;
    updateControls();
  }
});
$('cancel').addEventListener('click', () => {
  engine.cancelled = true;
  $('cancel').disabled = true;
  $('run-status').textContent = 'Cancelling after the current model operation…';
});
$('generate').addEventListener('click', async () => {
  $('error').hidden = true;
  let protein, settings;
  try {
    protein = parseSequence($('sequence').value);
    settings = { seed: Number($('seed').value), temperature: Number($('temperature').value) };
    if (!$('seed').value || !Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 4294967295) throw new Error('Seed must be an integer from 0 to 4,294,967,295.');
    if (settings.temperature < 0.1 || settings.temperature > 2 || !Number.isFinite(settings.temperature)) throw new Error('Temperature must be from 0.1 to 2.');
  } catch (error) {
    showError(error);
    return;
  }
  busy = true;
  updateControls();
  clearResult();
  $('results-description')?.replaceChildren('Prediction in progress. No generated image is available yet.');
  $('cancel').hidden = false;
  $('cancel').disabled = false;
  $('run-progress').hidden = false;
  $('run-progress').value = 0;
  $('run-status').classList.remove('ready');
  try {
    engine.reference = selectedReference;
    const output = await engine.predict(protein.sequence, settings, value => { $('run-progress').value = value; });
    const pixels = normalizePixels(output.pixels);
    const context = $('prediction').getContext('2d');
    const image = context.createImageData(256, 256);
    for (let i = 0; i < pixels.length; i++) {
      image.data[i * 4] = 0;
      image.data[i * 4 + 1] = pixels[i];
      image.data[i * 4 + 2] = 0;
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    result = { ...output, ...protein, settings, reference: selectedReference, images: new Map(referenceImages), created: new Date().toISOString() };
    $('prediction').hidden = false;
    $('prediction-placeholder').hidden = true;
    $('downloads').hidden = false;
    $('results-description')?.replaceChildren(`Model prediction conditioned on your sequence and ${selectedReference.cellLine} image ${selectedReference.index}. The landmarks contain no measured protein channel.`);
    $('prediction-caption').textContent = `Green: predicted protein · ${output.model}`;
    $('result-meta').textContent = `${protein.sequence.length} residues · ${selectedReference.cellLine} / image ${selectedReference.index} · ${output.model} ${output.precision} · seed ${settings.seed} · ${output.seconds.toFixed(1)} s`;
    $('run-status').textContent = 'Prediction complete. Results are ready below.';
    $('run-status').classList.add('ready');
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    $('run-status').textContent = error.name === 'AbortError' ? 'Prediction cancelled. No result was saved.' : 'Prediction failed. No generated result is available.';
    $('results-description')?.replaceChildren('No prediction is available. Review the status above before trying again.');
    if (error.name !== 'AbortError') showError(error);
  } finally {
    busy = false;
    $('cancel').hidden = true;
    $('run-progress').hidden = true;
    updateControls();
  }
});
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_');
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('download-image').addEventListener('click', () => {
  if (!result) return;
  const snapshot = result;
  $('prediction').toBlob(blob => {
    if (!blob) return showError(new Error('Unable to encode the PNG image.'));
    download(blob, `${snapshot.model}-${snapshot.reference.cellLine}-image-${snapshot.reference.index}-seed-${snapshot.settings.seed}.png`);
  });
});
$('download-zip').addEventListener('click', async () => {
  if (!result) return;
  const snapshot = result;
  $('download-zip').disabled = true;
  try {
    const { zipSync } = await import('./vendor/fflate.js');
    const files = Object.fromEntries(snapshot.images);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const context = canvas.getContext('2d');
    const pixels = normalizePixels(snapshot.pixels);
    for (const [name, green] of [['predicted-protein-green.png', true], ['predicted-protein-grayscale.png', false]]) {
      const image = context.createImageData(256, 256);
      for (let i = 0; i < pixels.length; i++) {
        image.data.set([green ? 0 : pixels[i], pixels[i], green ? 0 : pixels[i], 255], i * 4);
      }
      context.putImageData(image, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve));
      if (!blob) throw new Error('Unable to encode the predicted image.');
      files[name] = new Uint8Array(await blob.arrayBuffer());
    }
    download(new Blob([zipSync(files)], { type: 'application/zip' }), `${snapshot.model}-${snapshot.reference.cellLine}-image-${snapshot.reference.index}-seed-${snapshot.settings.seed}.zip`);
  } catch (error) { showError(error); }
  finally { $('download-zip').disabled = false; }
});
void initializeReferences();

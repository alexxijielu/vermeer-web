export const EXAMPLE_SEQUENCE = 'MEAPAQLLFLLLLWLPDTTGEIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASIRATGIPARFSGSGSGTEFTLTISILQSEDFAVYYCQQYNNWP';

export function variantFiles(manifest, variant) {
  const model = manifest.variants[variant];
  if (!model) throw new Error(`Vermeer-${variant} is not exported. Prepare that model before setup.`);
  return [...manifest.commonModels, model.graph, ...model.externalData, ...(manifest.notices ?? [])];
}

export function uniprotURL(identifier) {
  const value = identifier.trim().toUpperCase();
  if (!/^[A-Z0-9][A-Z0-9_]{1,19}(?:-[1-9][0-9]*)?$/.test(value)) {
    throw new Error('Enter a UniProt accession or entry name, for example P04637 or P53_HUMAN—not a sequence or URL.');
  }
  return `https://rest.uniprot.org/uniprotkb/${encodeURIComponent(value)}.fasta`;
}

export async function lookupUniProt(identifier, fetcher = fetch) {
  const response = await fetcher(uniprotURL(identifier), { signal: AbortSignal.timeout(20000) });
  if (!response.ok) {
    if (response.status === 404 || response.status === 400) throw new Error('No UniProt entry found. Use an accession (P04637) or entry name (P53_HUMAN), not a gene symbol.');
    throw new Error(`UniProt lookup failed (HTTP ${response.status}). Please try again later.`);
  }
  const fasta = (await response.text()).trim();
  if (!fasta.startsWith('>')) throw new Error('UniProt returned an unexpected response, not a FASTA record.');
  try {
    parseSequence(fasta);
  } catch (error) {
    throw new Error(`UniProt entry found, but its sequence cannot be used: ${error.message}`, { cause: error });
  }
  return fasta;
}

export function parseSequence(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.filter(line => line.trim().startsWith('>')).length > 1) {
    throw new Error('Please provide one protein at a time, not multiple FASTA records.');
  }
  const header = lines[0]?.trim().startsWith('>') ? lines.shift().trim().slice(1) : '';
  const sequence = lines.join('').replace(/\s/g, '').toUpperCase();
  if (!sequence) throw new Error('Enter a protein sequence first.');
  const invalid = [...new Set(sequence.replace(/[ACDEFGHIKLMNPQRSTVWYX]/g, ''))];
  if (invalid.length) throw new Error(`Unsupported sequence character(s): ${invalid.join(', ')}. Use amino-acid letters; remove gaps and stop symbols.`);
  if (sequence.length < 3 || sequence.length > 1022) {
    throw new Error(`This sequence has ${sequence.length.toLocaleString('en-US')} residues. This web application accepts 3–1,022 residues. The sequence will not be truncated.`);
  }
  return { sequence, name: header || 'unnamed_protein' };
}

export function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleToken(logits, temperature, random, topK = 2000) {
  if (!(temperature > 0) || !Number.isInteger(topK) || topK < 1) throw new Error('Invalid sampling settings.');
  const sorted = Array.from(logits, (value, index) => ({ value, index }));
  if (sorted.some(item => !Number.isFinite(item.value))) throw new Error('Model returned non-finite logits; prediction stopped.');
  sorted.sort((a, b) => b.value - a.value);
  const choices = sorted.slice(0, topK);
  if (!choices.length) throw new Error('Model returned empty logits.');
  const weights = choices.map(item => Math.exp((item.value - choices[0].value) / temperature));
  let threshold = random() * weights.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < weights.length; i++) {
    threshold -= weights[i];
    if (threshold < 0) return choices[i].index;
  }
  return choices.at(-1).index;
}

export function sampleProteinToken(logits, step, temperature, random, topK = 2000) {
  if (!Number.isInteger(step) || step < 0 || step > 258) throw new Error('Invalid protein-channel token position.');
  if (logits.length !== 16393) throw new Error('Model returned an unexpected token vocabulary; prediction stopped.');
  if (logits.some(value => !Number.isFinite(value))) throw new Error('Model returned non-finite logits; prediction stopped.');
  const start = step === 0 ? 16387 : step === 257 ? 16391 : step === 258 ? 16392 : 0;
  const end = step === 0 || step >= 257 ? start + 1 : 16384;
  // Restrict the vocabulary before top-k; singleton boundaries still consume one seeded draw.
  return start + sampleToken(logits.slice(start, end), temperature, random, topK);
}

export function normalizePixels(values) {
  if (values.length !== 256 * 256) throw new Error('Expected a 256 × 256 protein channel.');
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error('Decoder returned non-finite image values.');
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return Uint8ClampedArray.from(values, value => max === min ? 0 : Math.round((value - min) / (max - min) * 255));
}

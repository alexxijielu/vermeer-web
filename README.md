# Vermeer browser prototype

A standalone, local-first research prototype with a plain, accessible interface inspired by YeastSpotter. It runs actual Vermeer-B/L/XL, ESM-C-600M and VQ decoding in the browser through ONNX Runtime WebGPU. There is no inference server or simulated-image fallback.

## Public deployment

- Website source: https://github.com/alexxijielu/vermeer-web
- Browser-ready models: https://huggingface.co/alexijielu/vermeer-web
- Initial website address: https://alexxijielu.github.io/vermeer-web/
- Intended custom domain: https://vermeer.alexluresearch.com

The GitHub repository does not contain model weights, Python environments,
downloaded training metadata, or credentials. Public builds include the
reference PNG/token pack, pinned ONNX Runtime WebGPU files, fflate, and their
license notices. There is no inference backend.

```powershell
npm ci
npm test
npm run build
```

The build produces `dist`, a static site below GitHub Pages' 1 GB limit.
`deployment.json` selects the public Hugging Face model repository and an
immutable, verified commit revision. Builds reject missing/unpinned revisions.
Models download directly from Hugging Face only when the user explicitly
requests a local download; reference assets come from the website. Both are
SHA-256 verified before saving. Model license notices accompany the local
download. Selecting a prepared folder and predicting never fetch missing
model/reference files.

The Pages workflow tests and builds each push to `main`. Relative resource URLs
support both the GitHub project subpath and a custom-domain root. Runtime WASM
uses one thread, so the public static site does not require the development
server's cross-origin-isolation headers.

The page includes Google Analytics 4 measurement ID `G-DDN4WNM253` and a visible
analytics disclosure. No custom analytics events send protein sequences, images,
or local folder contents. Review Analytics enhanced-measurement settings and
applicable consent requirements before public launch.

To connect the custom domain, add a DNS CNAME named `vermeer` pointing to
`alexxijielu.github.io` at the domain provider, then configure
`vermeer.alexluresearch.com` in the repository's Pages settings and enforce
HTTPS once its certificate is provisioned. The apex domain need not change.
Until that DNS/configuration is complete, use the initial GitHub Pages URL.

## Run

Page copy is editable in `web/index.html`. Preserve IDs on interactive controls.
The optional `results-description` and `reference-sources` text blocks may be
removed without breaking folder selection or prediction. `npm test` checks
literal required DOM bindings against the page to catch accidental removals.

Node.js 24 and a current WebGPU-capable Chrome or Edge browser are recommended.

```powershell
npm ci
npm start
```

Open http://127.0.0.1:5173 and choose B, L or XL. **All models and references use one local data folder**, not browser-managed model storage:

1. Choose **Download / update local files** and select a folder to keep on disk. This saves ESM-C, the decoder, the selected variant, and the complete 41-cell-type reference pack automatically. It verifies existing files and downloads only missing or changed/corrupt files; rerun it to resume an interrupted download. One progress bar covers both models and references.
2. Choose **Set up model** to verify and initialize all components directly from that folder.
3. On later visits, choose **Choose existing local folder**, select the same folder and run setup again. No model or reference asset downloads or browser-cache copies are made. Folder access uses the File System Access API in Chrome/Edge on localhost or HTTPS; folder permission must be granted again for each visit. Small manifests are still read from the application to verify local files.

Read-only fields show the selected folder name and `<selected-folder>\references`. Browsers expose a directory handle's name, not its absolute OS path; these are explicitly labeled relative paths.

Existing local model folders remain compatible: choose **Download / update local files** and select the old folder to add the reference pack without downloading valid weights again. For this working copy, select `web/assets/models` with that action; reference files will be added there from the project's prepared assets. Do not select its parent or its weight-shard subfolders.

There is no browser-cache alternative or separate reference location. Switching models requires setup again and can reuse the same local folder and shared ESM-C/decoder files; use Download / update to add the chosen variant. Setup and prediction never fetch missing weights or reference images automatically. Missing, modified or inaccessible files produce an explicit error. Updates are verified against the application's current SHA-256 manifests.

After setup, enter a sequence, choose a reference field, then choose **Predict protein localization**. Optional UniProt lookup retrieves a sequence by accession or entry name. It sends only that identifier to `rest.uniprot.org`, not a pasted sequence. Entries outside the supported 3–1,022-residue range report the retrieved sequence length and leave the existing input unchanged; sequences are never automatically truncated. The local server only serves static files. Do not expose this development server publicly.

### Local reference images

The reference collection covers all 41 cell lines / cell types in the pinned HPA metadata, with ten distinct fields each (410 total). Every model download includes all 2,051 reference PNG/JSON assets, totaling 91,174,738 bytes (91.2 MB). Reference images and VQ-token JSONs live in the `references` subfolder; the small `references.json` catalog sits beside the model files at the selected folder's root. No raw TIFFs are saved.

Choosing the existing local data folder also loads its reference catalog and preview. The app reads and verifies only the selected field's JSON and four PNGs directly from disk, without network fallback or Cache Storage copies. No reference images are fetched before selecting a folder.

The browser retains only the active field and releases old image URLs / image bytes when selection changes; superseded loads are cancelled. Growing the collection does not require loading all images into RAM. Disk storage reduces repeat downloads, not model inference RAM.

Exported model files are already present in this working copy. To regenerate them, use Python 3.14 and:

```powershell
.\tools\prepare.ps1
```

This creates an isolated conversion environment, installs pinned dependencies, downloads the published image-model checkpoints and ESM-C checkpoint, and exports and checks the models. Several GB of disk space and a substantial amount of RAM are needed for conversion. Conversion is a one-time developer operation, not a browser requirement.

## Scope

- Ten distinct real fields for each of the 41 HPA cell lines / cell types, sourced only from original TIFF nucleus, microtubule and ER channels at native 8-bit or 16-bit depth. The measured protein/green channel is never downloaded. A dropdown selects image 1-10, and a landmark-only preview is shown before prediction.
- Arbitrary single protein sequences or single-record FASTA, 3-1,022 residues.
- Vermeer-B/L/XL, FP32 throughout, 256 x 256 output, batch size one, top-k 2,000, guidance 1. Larger models use more memory and are slower; higher capacity is not a guarantee of better prediction for every sequence.
- Seed and temperature controls and cancellation between model operations.
- Position-constrained protein-channel sampling: the start marker (16387) is the only allowed token at position 0, image-code tokens (0–16383) are allowed at positions 1–256, the end marker (16391) at position 257, and EOS (16392) at position 258. Constraints are applied before top-k and temperature-based sampling; invalid image tokens are never clamped into the codebook. Each position consumes one seeded random draw, including forced markers. Non-finite model logits and invalid vocabulary sizes still stop prediction.
- A predicted green PNG, plus an image ZIP containing three separate grayscale landmark PNGs, their color composite, and predicted green and grayscale protein PNGs. No settings JSON is included.
- On-device ESM-C embedding, loaded one layer at a time to bound model residency. Up to four embeddings are cached in this tab.
- Per-layer GPU-resident Vermeer attention caches during autoregressive generation. This avoids packing the larger variants' entire cache into one WebGPU storage buffer.
- SHA-256 and byte-length checks on model files.

This is deliberately a correctness-first FP32 prototype, not the optimized FP16 deployment discussed during feasibility assessment. Setup verifies ESM-C, the decoder and the selected Vermeer variant from the chosen local folder and checks their inference sessions before enabling prediction. Model files total approximately 2.913 GB (B), 3.843 GB (L), or 5.573 GB (XL); including the reference pack, downloads total approximately 3.00, 3.93, or 5.66 GB. Switching variants reuses shared files. Each prediction reads prepared local files and activates/releases GPU stages sequentially; it never fetches missing model weights. Folder reads also verify hashes to detect files modified after setup. Browser-cache eviction does not affect a user-owned data folder. Disk-folder storage does not reduce model size or inference RAM requirements. XL may exceed a browser/device's available memory.

Transformer graphs use external weight shards to avoid a multi-GB single JavaScript buffer/protobuf. Reference fields are prepared with `tools/prepare_references.py`: only original `.tif.gz` sources are accepted; no JPEG fallback exists. TIFFs are downloaded and decoded transiently in memory, never written to disk. Preparation uses bounded download prefetch and batched three-channel VQ encoding; complete verified fields are reused when resuming. TIFF pages are read as native grayscale samples without palette rendering or external OME-series expansion.

The image steps match paper Appendix A.4 and the repository's extraction path: LANCZOS resize to 256 x 256; 1st/99th-percentile `rescale_intensity`; divide by the channel maximum and cast float32; multiply by 255 and cast uint8; replicate grayscale to RGB; normalize to [-1,1]; VQ encode. All-zero channels remain zero, as in the upstream normalization helper. Inference uses a single unaugmented view, consistent with validation extraction, not the training TenCrop / flip / rotation bank described in Appendix A.4.2.

The saved PNGs preserve the exact quantized tokenizer inputs without additional lossy compression. This is not a raw microscopy archive: changing normalization or reprocessing at higher precision requires downloading the TIFFs again. Source URLs, compressed-source SHA-256 hashes, native shape/dtype and percentile values remain in each field JSON. `web/assets/reference-manifest.json` records every prepared asset's hash/size, the pinned metadata revision, the tokenizer checkpoint hash and the preprocessing convention. No raw TIFFs are included in the browser pack.

Landmark images credit the Human Protein Atlas and Thul et al., *Science* (2017), https://doi.org/10.1126/science.aal3321. HPA data is [CC BY 4.0](https://www.proteinatlas.org/about/licence). Each landmark PNG embeds its attribution, specific source URL, citation and processing description, so these travel with the image-only ZIP.

## Scientific caveats

Generated images are hypotheses, not observed or calibrated protein fluorescence. Browser numerical parity does not validate biological fidelity. Providing reference fields for every metadata cell type does not establish equal accuracy across those cell types or demonstrate that each was represented in model training. Model-weight quantization, robust performance on 16 GB machines, and validation against the paper's biological metrics remain follow-up work.

**Pooling compatibility:** the repository's `prepare_protein_prefix.py` removes special tokens with `[1:-1]`, then its mean-pooling helper applies `[1:-1]` again. This prototype deliberately reproduces that double trim (excluding the first and last actual residue from the pooling average) rather than silently changing the conditioning convention. All residues still participate in the ESM-C attention computation. Confirm the intended preprocessing with the model authors before scientific use.

The current, pinned Biohub ESM-C checkpoint is loaded via the Transformers ESM-C implementation. Its exported computation is checked against that implementation; equivalence to the original training-time ESM package and historical weights is not established. The included published embedding is retained in the reference JSON for comparison but is **not** silently substituted for an arbitrary input sequence.

The selectable reference PNGs are the exact normalized 256 x 256 TIFF-derived landmark inputs used for VQ encoding, not JPEGs or decoded approximations. Their presentation maps nucleus to blue, microtubules to red and ER to yellow. There is no measured protein in the composite. Predicted protein is displayed in green, independently of the landmark image. Display normalization is for visualization, not calibrated intensity analysis. The original single repository example is retained separately for numerical parity fixtures.

## Validation

The unified local download saved all 130 B/common-model files plus all 2,051 reference assets into one real browser directory handle: 2,181 files, 3,004,167,763 bytes, and zero TIFFs. After reloading, choosing that folder loaded all 41 cell types and ten field options without asset requests. Repeating the combined download fetched zero model/reference assets and made zero Cache Storage calls. Cancelling folder selection preserved the previous folder and preview; an empty folder produced an explicit missing-reference error without fetching images. Browser automation substitutes a temporary OPFS handle for the native picker; it does not validate the native OS dialog.

Unified-folder setup and a real HUVEC/TERT2 image-10 B prediction both passed with zero model/reference requests and zero Cache Storage calls; prediction made zero fetch requests of any kind. The output was 256 x 256 with nonzero green and zero red/blue. Both read-only path fields fit a 375-pixel viewport without horizontal overflow. Temporary browser test files were removed after validation.

The complete prepared collection also passed six Python tests covering every field's hashes, PNGs, composite colors, provenance and token boundaries, with representative exact VQ code reproduction. All 410 browser previews were exercised with at most one live reference image object URL. Before position-constrained sampling was introduced, a sampled U2OS generation emitted an invalid channel-boundary token and was explicitly rejected rather than replaced. The current sampler prevents boundary tokens in image-code positions. This intentionally differs from the pinned upstream sampler, which samples without positional constraints and clamps out-of-range image tokens after generation ([upstream decoding](https://github.com/microsoft/vermeer/blob/5c4ac7f1eeb722f1a8fe68c48743dc6a197806bf/vermeer/autoregressive/models/generate_ca.py#L457)). Model weights are unchanged, but results for a given seed can differ from earlier browser versions and upstream generation.

```powershell
npm test
.\tools\.venv\Scripts\python.exe -m py_compile .\tools\export_models.py
.\tools\.venv\Scripts\python.exe .\tools\test_references.py
```

Conversion fails if numerical comparisons exceed tolerances. Reports are written to `web/assets/vermeer-{b,l,xl}-parity.json`, `web/assets/image-parity.json` and `web/assets/esm-parity.json`. Checks include Vermeer prefill against the upstream model, three incremental cached steps for each variant, ONNX/PyTorch decoder agreement, all 36 ESM-C layers, and assembled PyTorch ESM-C embedding agreement.

`web/assets/manifest.json` records hashes, sizes and source revisions. Native CPU ONNX parity alone is not proof of browser GPU parity; end-to-end browser checks are also required.

Local-folder work passes 20 JavaScript tests, covering direct verified reads without Cache Storage/network fallback, combined model/reference downloads and progress, selected-field-only loading, cancellation, resumable downloads, shared-file reuse across variants, corruption repair, permission errors, write/cleanup failures and file-size checks before allocation. Explicit folder downloads use `cache: 'no-store'` to avoid an additional HTTP-cache copy. A browser file-API check saved all 130 B files, fetched zero files on a repeat download, and completed a real 115-residue B prediction in 36.9 seconds with zero model-file requests during setup or prediction and zero Cache Storage calls. Automation supplied a temporary OPFS `FileSystemDirectoryHandle` instead of interacting with the native OS folder picker; this exercises the actual disk read/write APIs and inference path, not the system dialog or a user-owned folder's durability after clearing browser storage. Production uses only the user-selected folder, not OPFS.

A subsequent browser pass reopened the folder after reload, rejected a deliberately corrupted graph, downloaded only that graph with `cache: 'no-store'`, and successfully reran setup. Selecting an empty folder produced an explicit missing-file error without a model request. The layout had no horizontal overflow at 375 pixels, and all temporary test files were removed. The former cache alternative has since been removed; inference now requires a local directory handle.

The expanded prototype completed real browser predictions for UniProt P04637 (393 residues), Rh30 image 10, seed 42, on the AMD Radeon 860M / 63.3 GB RAM Windows machine:

| Variant | Prediction time | Model-file network requests during prediction |
| --- | ---: | ---: |
| B | 26.4 s | 0 |
| L | 33.6 s | 0 |
| XL | 104.0 s | 0 |

These exclude setup. L and XL reused the in-tab sequence embedding, so the numbers are functional checks, not a controlled speed comparison or evidence of support on 16 GB laptops.

Eight JavaScript tests and two TIFF reference tests passed. The reference checks cover ten unique source hashes and token prefixes, original uint16 TIFFs, exact normalized landmark pixels, channel boundaries, composite colors, embedded attribution and JPEG rejection. Browser checks also passed for live UniProt lookup, preserving the sequence after invalid lookup input, all ten field options and preview switching, setup-required gating after model changes, and a six-PNG ZIP with a green-only predicted channel. A fresh-page check confirmed three original TIFF source links, no horizontal overflow at a 375-pixel viewport, and successful preparation of all B/ESM-C/decoder components before prediction.

Initial single-model browser checks passed on an AMD Radeon 860M integrated GPU in a Windows machine with 63.3 GB RAM. A 115-residue sequence produced a real 256 x 256 result in 41.1 seconds, excluding initial setup. These historical measurements predate the multi-model/setup-cache changes. This is not a benchmark establishing support for 16 GB machines.

Independent browser/PyTorch comparisons passed for the full protein embedding (maximum absolute error `4.03e-7`), Vermeer prefill and three cached decoding steps (maximum `2.87e-5`), and all decoder pixels (maximum `1.73e-5`). Measurements are retained in `web/assets/browser-parity.json`.

To reproduce these GPU checks, generate fixtures:

```powershell
.\tools\.venv\Scripts\python.exe .\tools\make_browser_fixture.py
```

Then open the application and run this in the browser developer console:

```javascript
const directory = await showDirectoryPicker({ mode: 'read' });
await (await import('/verify.js')).verifyBrowser('B', directory);
```

This performs real inference and reads the prepared models from your selected folder; do not run it concurrently with another prediction.

For other variants, generate fixtures with `--variant L` or `--variant XL` and pass `'L'` or `'XL'` and the directory handle to `verifyBrowser`.

## Sources and attribution

- [Vermeer repository](https://github.com/microsoft/vermeer), revision `5c4ac7f1eeb722f1a8fe68c48743dc6a197806bf`.
- [Vermeer paper](https://www.biorxiv.org/content/10.64898/2026.06.01.729395v1).
- [Vermeer-B checkpoint](https://huggingface.co/microsoft/vermeer-B).
- [LlamaGen VQ tokenizer](https://huggingface.co/FoundationVision/LlamaGen).
- [ESM-C-600M](https://huggingface.co/biohub/ESMC-600M), revision `fcc9d36f5fff97d9bb69a36e3252de61cc78c968`.
- [YeastSpotter](http://yeastspotter.csb.utoronto.ca/) for interface inspiration only; no source or branding copied.

Upstream Vermeer code and its MIT license are preserved under `tools/upstream`. Review the relevant checkpoint and reference-data terms before redistributing model files or deploying publicly.

"""Prepare ten original-TIFF reference fields for every cell type in pinned HPA metadata."""

import csv
import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image, PngImagePlugin
from skimage.exposure import rescale_intensity

from export_models import OUTPUT, TOOLS, VQ_models, save_json

REVISION = "5c4ac7f1eeb722f1a8fe68c48743dc6a197806bf"
METADATA = f"https://raw.githubusercontent.com/microsoft/vermeer/{REVISION}/scripts/prepare_data/hpa_cell-line-download.csv"
CHANNELS = [("blue", "nucleus"), ("red", "microtubules"), ("yellow", "er")]
FIELDS_PER_LINE = 10
PROCESSING_VERSION = 2


def attribution(source: str) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Human Protein Atlas; https://www.proteinatlas.org")
    info.add_text("License", "CC BY 4.0; https://creativecommons.org/licenses/by/4.0/")
    info.add_text("Source", source)
    info.add_text("Citation", "Thul PJ et al. A subcellular map of the human proteome. Science (2017). https://doi.org/10.1126/science.aal3321")
    info.add_text("Processing", "Original HPA TIFF at native bit depth; LANCZOS resize to 256x256; 1st/99th-percentile rescale; uint8 display quantization; composite uses blue nucleus, red microtubules, yellow ER.")
    return info


def tiff_url(url: str) -> str:
    url = re.sub(r"_cr[a-f0-9]+_?(?=(?:red|blue|yellow)\.tif\.gz$)", "_", url)
    if not url.startswith("https://images.proteinatlas.org/") or not url.endswith(".tif.gz"):
        raise ValueError(f"Expected an original HPA TIFF URL, not a JPEG or other source: {url}")
    return url


def cell_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not slug:
        raise ValueError(f"Cell type has no safe directory name: {name!r}")
    return slug


def select_fields(records):
    selected = {}
    seen = {}
    for row in records:
        name = row["cell_line"]
        fields = selected.setdefault(name, [])
        if len(fields) == FIELDS_PER_LINE:
            continue
        urls = {color: tiff_url(row[f"{color}_imageUrl"]) for color, _ in CHANNELS}
        corrections = {}
        if urls == {
            "blue": "https://images.proteinatlas.org/4109/23_H11_1_blue.tif.gz",
            "red": "https://images.proteinatlas.org/4109/1843_B2_17_red.tif.gz",
            "yellow": "https://images.proteinatlas.org/4109/23_H11_1_yellow.tif.gz",
        }:
            urls["red"] = "https://images.proteinatlas.org/4109/23_H11_1_red.tif.gz"
            corrections["red"] = {
                "metadataUrl": row["red_imageUrl"], "usedUrl": urls["red"],
                "reason": "The pinned metadata pairs a different field's microtubules with nucleus/ER from 23_H11_1; use the original TIFF from the matching field.",
            }
        stems = {re.sub(r"_(?:blue|red|yellow)\.tif\.gz$", "", url) for url in urls.values()}
        if len(stems) != 1:
            raise ValueError(f"Landmark channels refer to different fields for {name}: {urls}")
        unique = seen.setdefault(name, set())
        if urls["blue"] in unique:
            continue
        unique.add(urls["blue"])
        field = {
            "cellLine": name, "antibody": row["antibody"],
            "uniprot": row["uniprot"], "fov": row["fov"], "sources": urls,
        }
        if corrections:
            field["sourceCorrections"] = corrections
        fields.append(field)
    short = {name: len(fields) for name, fields in selected.items() if len(fields) != FIELDS_PER_LINE}
    if short:
        raise ValueError(f"Not enough distinct TIFF fields for every cell type: {short}")
    slugs = [cell_slug(name) for name in selected]
    if len(set(slug.casefold() for slug in slugs)) != len(slugs):
        raise ValueError("Cell type names collide on a case-insensitive filesystem.")
    return dict(sorted(selected.items(), key=lambda item: item[0].casefold()))


def select_rows():
    metadata = TOOLS / "hpa-cell-line-download.csv"
    if not metadata.exists():
        with urllib.request.urlopen(METADATA, timeout=90) as response:
            data = response.read()
        metadata.write_bytes(data)
    with metadata.open(encoding="utf-8", newline="") as stream:
        rows = select_fields(csv.DictReader(stream))
    save_json(TOOLS / "reference-sources-all.json", {"metadataRevision": REVISION, "cellLines": rows})
    return rows


def download_channel(row, index, color):
    url = tiff_url(row["sources"][color])
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            return data
        except (urllib.error.URLError, TimeoutError) as error:
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code == 429 or error.code >= 500
            if not retryable or attempt == 4:
                raise
            print(f"Retrying TIFF {row['cellLine']} / {index} / {color}: {error}", flush=True)
            time.sleep(2 ** (attempt + 1))


def read_channel(row, index, color):
    url = tiff_url(row["sources"][color])
    raw = download_channel(row, index, color)
    with tifffile.TiffFile(io.BytesIO(gzip.decompress(raw))) as image:
        if len(image.pages) != 1:
            raise ValueError(f"Expected a single-page landmark TIFF: {url}")
        original = image.pages[0].asarray()
        if original.ndim != 2 or original.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError(f"Expected a single-channel 8-bit or 16-bit TIFF: {url}")
        resized = np.array(Image.fromarray(original).resize((256, 256), Image.Resampling.LANCZOS))
    p1, p99 = np.percentile(resized, (1, 99))
    if np.max(resized) == 0:
        normalized = resized.astype(np.float32)
    else:
        scaled = rescale_intensity(resized, in_range=(p1, p99))
        normalized = (scaled / np.max(scaled)).astype(np.float32)
    pixels = (normalized * 255).astype(np.uint8)
    return pixels, {
        "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
        "originalShape": list(original.shape), "originalDtype": str(original.dtype),
        "reader": "tifffile single-page raw samples; no palette rendering or OME-series expansion",
        "resizedPercentiles": [float(p1), float(p99)],
    }


def file_entry(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def prepared_reference(row, index):
    path = OUTPUT / "references" / cell_slug(row["cellLine"]) / f"{index:02}.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference["sources"] != row["sources"]:
        return None
    if reference.get("processingVersion") is None and row["cellLine"] == "Rh30":
        if len(reference["prefix"]) != 774 or not all((OUTPUT / name).is_file() for name in reference["images"]):
            raise ValueError(f"Incomplete legacy reference: {path}")
        reference["processingVersion"] = PROCESSING_VERSION
        reference["files"] = {name: file_entry(OUTPUT / name) for name in reference["images"]}
        save_json(path, reference)
    if reference.get("processingVersion") != PROCESSING_VERSION:
        return None
    for name, expected in reference["files"].items():
        source = OUTPUT / name
        if not source.exists() or file_entry(source) != expected:
            return None
    return reference


@torch.inference_mode()
def prepare():
    rows_by_line = select_rows()
    vq = VQ_models["VQ-16"]().eval()
    vq.load_state_dict(
        torch.load(TOOLS / "checkpoints" / "vq_ds16_c2i.pt", map_location="cpu", weights_only=True)["model"],
        strict=True,
    )
    catalog = {"version": 2, "metadataRevision": REVISION, "fieldsPerCellLine": FIELDS_PER_LINE, "cellLines": {}}
    files = {}
    browser_files = []
    jobs = [(index, row) for rows in rows_by_line.values() for index, row in enumerate(rows, 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        iterator = iter(jobs)

        def schedule(job):
            index, row = job
            reference = prepared_reference(row, index)
            futures = [] if reference else [pool.submit(read_channel, row, index, color) for color, _ in CHANNELS]
            return index, row, reference, futures

        pending = deque(schedule(job) for job in [next(iterator, None) for _ in range(4)] if job is not None)
        completed = 0
        while pending:
            index, row, reference, futures = pending.popleft()
            following = next(iterator, None)
            if following is not None:
                pending.append(schedule(following))
            name = row["cellLine"]
            slug = cell_slug(name)
            if reference is None:
                results = [future.result() for future in futures]
                reference = encode_reference(vq, row, index, results)
            json_path = OUTPUT / "references" / slug / f"{index:02}.json"
            json_name = json_path.relative_to(OUTPUT).as_posix()
            files[json_name] = file_entry(json_path)
            browser_files.append(json_name)
            for path, entry in reference["files"].items():
                files[path] = entry
                browser_files.append(path)
            catalog["cellLines"].setdefault(name, []).append({
                "index": index, "id": reference["id"], "url": f"/assets/{json_name}",
            })
            completed += 1
            print(f"Prepared TIFF field {completed}/{len(jobs)}: {name} / {index}", flush=True)
    catalog_path = OUTPUT / "references.json"
    save_json(catalog_path, catalog)
    files["references.json"] = file_entry(catalog_path)
    browser_files.insert(0, "references.json")
    save_json(OUTPUT / "reference-manifest.json", {
        "version": 1, "metadataRevision": REVISION,
        "cellLineCount": len(rows_by_line), "fieldCount": len(jobs),
        "files": files, "browserFiles": browser_files,
        "browserBytes": sum(files[path]["bytes"] for path in browser_files),
        "tokenizer": {"name": "LlamaGen VQ-16", "checkpointSha256": file_entry(TOOLS / "checkpoints" / "vq_ds16_c2i.pt")["sha256"]},
        "preprocessing": {
            "resize": "PIL LANCZOS to 256x256 before intensity normalization",
            "intensity": "1st/99th percentile rescale_intensity, divide by channel maximum, float32; preserve all-zero channels",
            "tokenizerInput": "multiply by 255, uint8 quantization, RGB replication, normalize to [-1,1]",
            "view": "single unaugmented inference/validation view; no training TenCrop/flip/rotation bank",
            "sourceStorage": "transient original TIFF bytes only; retain PNGs, tokens and source SHA-256 provenance",
        },
    })
    print(f"Published {len(jobs)} TIFF-derived fields across {len(rows_by_line)} cell types.", flush=True)


def encode_reference(vq, row, index, results):
    prefix = []
    relative = Path("references") / cell_slug(row["cellLine"]) / f"{index:02}"
    folder = OUTPUT / relative
    folder.mkdir(parents=True, exist_ok=True)
    images = [(relative / "reference.png").as_posix()]
    provenance = {}
    tensors = [
        torch.from_numpy(pixels.astype(np.float32) / 127.5 - 1).reshape(1, 1, 256, 256).repeat(1, 3, 1, 1)
        for pixels, _ in results
    ]
    _, _, info = vq.encode(torch.cat(tensors))
    all_codes = info[2].reshape(3, 256).numpy()
    for channel, ((color, name), (pixels, source)) in enumerate(zip(CHANNELS, results)):
        filename = f"landmark-{name}.png"
        Image.fromarray(pixels).save(folder / filename, pnginfo=attribution(source["url"]))
        images.append((relative / filename).as_posix())
        provenance[name] = source
        codes = all_codes[channel]
        if codes.shape != (256,) or codes.min() < 0 or codes.max() >= 16384:
            raise ValueError("VQ encoder produced invalid landmark codes.")
        prefix.extend([16384 + channel, *codes.tolist(), 16388 + channel])
    blue, red, yellow = [result[0] for result in results]
    rgb = np.stack((np.maximum(red, yellow), yellow, blue), axis=-1)
    Image.fromarray(rgb).save(folder / "reference.png", pnginfo=attribution("; ".join(row["sources"].values())))
    reference = {
        **row, "id": f"{row['uniprot']}_{row['antibody']}_{cell_slug(row['cellLine'])}_fov_{row['fov']}",
        "index": index, "prefix": prefix, "images": images,
        "sourceFormat": "original HPA TIFF at native 8-bit or 16-bit depth, gzip-compressed; no JPEG fallback",
        "preprocessing": "LANCZOS resize to 256x256, 1st/99th-percentile rescale_intensity, normalize to maximum, uint8 quantization, RGB replication, scale to [-1,1], VQ-16 encode",
        "provenance": provenance, "metadataRevision": REVISION,
        "processingVersion": PROCESSING_VERSION,
    }
    reference["files"] = {path: file_entry(OUTPUT / path) for path in images}
    json_path = OUTPUT / "references" / cell_slug(row["cellLine"]) / f"{index:02}.json"
    save_json(json_path, reference)
    return reference


if __name__ == "__main__":
    prepare()

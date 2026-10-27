import gzip
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
import torch
from PIL import Image
from skimage.exposure import rescale_intensity

from prepare_references import OUTPUT, TOOLS, VQ_models, cell_slug, read_channel, select_rows, tiff_url


class ReferenceTests(unittest.TestCase):
    def test_ten_distinct_original_tiff_fields_and_model_inputs(self):
        catalog = json.loads((OUTPUT / "references.json").read_text())
        self.assertEqual(set(catalog["cellLines"]), set(select_rows()))
        self.assertEqual(len(catalog["cellLines"]), 41)
        expected_files = {"references.json"}
        for cell_line, fields in catalog["cellLines"].items():
            self.assertEqual([item["index"] for item in fields], list(range(1, 11)))
            seen = set()
            seen_hashes = set()
            for item in fields:
                relative = item["url"].removeprefix("/assets/")
                expected_files.add(relative)
                reference = json.loads((OUTPUT / relative).read_text())
                self.assertEqual(reference["cellLine"], cell_line)
                self.assertNotIn(reference["sources"]["blue"], seen)
                seen.add(reference["sources"]["blue"])
                self.assertEqual(len(reference["prefix"]), 774)
                blue_hash = reference["provenance"]["nucleus"]["sha256"]
                self.assertNotIn(blue_hash, seen_hashes)
                seen_hashes.add(blue_hash)
                images = {}
                for color, name in [("blue", "nucleus"), ("red", "microtubules"), ("yellow", "er")]:
                    source = reference["provenance"][name]
                    self.assertTrue(tiff_url(source["url"]).endswith(".tif.gz"))
                    self.assertIn(source["originalDtype"], ["uint8", "uint16"])
                    self.assertRegex(source["sha256"], r"^[a-f0-9]{64}$")
                    self.assertEqual(len(source["originalShape"]), 2)
                    path = OUTPUT / "references" / cell_slug(cell_line) / f"{item['index']:02}" / f"landmark-{name}.png"
                    with Image.open(path) as image:
                        self.assertEqual(image.size, (256, 256))
                        self.assertEqual(image.mode, "L")
                        self.assertIn("Human Protein Atlas", image.info["Author"])
                        self.assertIn(source["url"], image.info["Source"])
                        images[color] = np.array(image)
                self.assertNotIn("green", reference["sources"])
                for channel in range(3):
                    tokens = reference["prefix"][channel * 258:(channel + 1) * 258]
                    self.assertEqual(tokens[0], 16384 + channel)
                    self.assertEqual(tokens[-1], 16388 + channel)
                    self.assertTrue(all(isinstance(token, int) and 0 <= token < 16384 for token in tokens[1:-1]))
                with Image.open(OUTPUT / reference["images"][0]) as image:
                    composite = np.array(image)
                np.testing.assert_array_equal(composite[:, :, 0], np.maximum(images["red"], images["yellow"]))
                np.testing.assert_array_equal(composite[:, :, 1], images["yellow"])
                np.testing.assert_array_equal(composite[:, :, 2], images["blue"])
                expected_files.update(reference["images"])
        manifest = json.loads((OUTPUT / "reference-manifest.json").read_text())
        self.assertEqual(manifest["fieldCount"], 410)
        self.assertEqual(set(manifest["browserFiles"]), expected_files)
        self.assertEqual(set(manifest["files"]), expected_files)
        self.assertEqual(manifest["browserBytes"], sum(entry["bytes"] for entry in manifest["files"].values()))
        for name, entry in manifest["files"].items():
            self.assertTrue(name.endswith((".png", ".json")))
            raw = (OUTPUT / name).read_bytes()
            self.assertEqual(len(raw), entry["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])

    @torch.inference_mode()
    def test_saved_tokens_match_official_tokenizer_inputs(self):
        vq = VQ_models["VQ-16"]().eval()
        vq.load_state_dict(torch.load(TOOLS / "checkpoints" / "vq_ds16_c2i.pt", map_location="cpu", weights_only=True)["model"], strict=True)
        for cell_line, index in [("Rh30", 1), ("A-431", 3), ("HUVEC/TERT2", 10)]:
            reference = json.loads((OUTPUT / "references" / cell_slug(cell_line) / f"{index:02}.json").read_text())
            pixels = np.stack([np.array(Image.open(OUTPUT / name)) for name in reference["images"][1:]])
            inputs = torch.from_numpy(pixels.copy()).unsqueeze(1).repeat(1, 3, 1, 1).to(torch.float32).div(255).sub(0.5).div(0.5)
            _, _, info = vq.encode(inputs)
            expected = np.array([reference["prefix"][i * 258 + 1:i * 258 + 257] for i in range(3)])
            np.testing.assert_array_equal(info[2].reshape(3, 256).numpy(), expected)

    def test_jpeg_sources_are_never_accepted(self):
        with self.assertRaises(ValueError):
            tiff_url("https://images.proteinatlas.org/example.jpg")


class PreprocessingTests(unittest.TestCase):
    def test_native_tiff_preprocessing_matches_paper_and_repository_without_disk_writes(self):
        for dtype in [np.uint8, np.uint16]:
            for kind in ["texture", "zero", "constant"]:
                with self.subTest(dtype=dtype, kind=kind):
                    samples = np.random.default_rng(42).integers(0, np.iinfo(dtype).max, size=(512, 384), dtype=dtype)
                    if kind != "texture":
                        samples.fill(0 if kind == "zero" else 17)
                    buffer = io.BytesIO()
                    tifffile.imwrite(buffer, samples)
                    raw = gzip.compress(buffer.getvalue())
                    url = "https://images.proteinatlas.org/test/field_blue.tif.gz"
                    with patch("prepare_references.download_channel", return_value=raw), patch("pathlib.Path.write_bytes", side_effect=AssertionError("TIFFs must not be stored")):
                        actual, provenance = read_channel({"sources": {"blue": url}}, 1, "blue")
                    resized = np.array(Image.fromarray(samples).resize((256, 256), Image.Resampling.LANCZOS))
                    if np.max(resized) == 0:
                        normalized = resized
                    else:
                        limits = np.percentile(resized, (1, 99))
                        scaled = rescale_intensity(resized, in_range=tuple(limits))
                        normalized = (scaled / np.max(scaled)).astype(np.float32)
                    expected = (normalized * 255).astype(np.uint8)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(provenance["originalDtype"], str(np.dtype(dtype)))
                    self.assertEqual(provenance["sha256"], hashlib.sha256(raw).hexdigest())

    def test_native_tiff_page_reader_preserves_uint16_palette_samples(self):
        samples = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
        palette = np.tile(np.arange(65536, dtype=np.uint16), (3, 1))
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, samples, photometric="palette", colormap=palette)
        with patch("prepare_references.download_channel", return_value=gzip.compress(buffer.getvalue())):
            pixels, provenance = read_channel({"sources": {"blue": "https://images.proteinatlas.org/test/palette_blue.tif.gz"}}, 1, "blue")
        self.assertEqual(pixels.shape, (256, 256))
        self.assertEqual(provenance["originalDtype"], "uint16")

    def test_filesystem_names_preserve_distinct_cell_types_without_path_separators(self):
        names = list(select_rows())
        slugs = [cell_slug(name) for name in names]
        self.assertEqual(len(set(slug.casefold() for slug in slugs)), len(names))
        self.assertTrue(all("/" not in slug and "\\" not in slug for slug in slugs))
        self.assertEqual(cell_slug("HUVEC/TERT2"), "HUVEC_TERT2")


if __name__ == "__main__":
    unittest.main()

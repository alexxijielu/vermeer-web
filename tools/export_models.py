"""Export real FP32 checkpoints, checking ONNX against the PyTorch implementations."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
OUTPUT = ROOT / "web" / "assets"
MODELS = OUTPUT / "models"
sys.path.insert(0, str(TOOLS / "upstream"))

from autoregressive.models.gpt_ca import GPT_models, apply_rotary_emb
from tokenizer.tokenizer_image.vq_model import VQ_models

torch.set_num_threads(6)
torch.set_grad_enabled(False)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def export_checked(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
    name: str,
    inputs: list[str],
    outputs: list[str],
    axes: dict[str, dict[int, str]] | None = None,
    atol: float = 0.001,
    external_weights: bool = False,
) -> dict[str, float]:
    model.eval()
    destination = MODELS / name
    torch.onnx.export(
        model, args, str(destination), input_names=inputs, output_names=outputs,
        dynamic_axes=axes, opset_version=17, dynamo=False,
        export_params=not external_weights,
        do_constant_folding=not external_weights,
    )
    if external_weights:
        graph = onnx.load(str(destination))
        tensors = dict(model.named_parameters()) | dict(model.named_buffers())
        weights_dir = MODELS / f"{destination.stem}-weights"
        weights_dir.mkdir(exist_ok=True)
        for item in list(graph.graph.input):
            if item.name in inputs:
                continue
            if item.name not in tensors:
                raise ValueError(f"Unknown exported parameter: {item.name}")
            value = tensors[item.name].detach().contiguous().numpy()
            filename = f"{len(graph.graph.initializer):03}.bin"
            value.tofile(weights_dir / filename)
            initializer = onnx.TensorProto()
            initializer.name = item.name
            initializer.data_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            initializer.dims.extend(value.shape)
            initializer.data_location = onnx.TensorProto.EXTERNAL
            for key, val in {
                "location": f"{weights_dir.name}/{filename}",
                "offset": "0", "length": str(value.nbytes),
            }.items():
                entry = initializer.external_data.add()
                entry.key, entry.value = key, val
            graph.graph.initializer.append(initializer)
            graph.graph.input.remove(item)
        onnx.save(graph, str(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual = session.run(None, dict(zip(inputs, [arg.numpy() for arg in args])))
    expected = model(*args)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    errors = {}
    for key, result, reference in zip(outputs, actual, expected):
        baseline = reference.numpy()
        np.testing.assert_allclose(result, baseline, atol=atol, rtol=0.001)
        errors[key] = float(np.max(np.abs(result - baseline)))
    print(f"Validated {name}: {errors}", flush=True)
    del session
    return errors


class BrowserGPT(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.heads = model.config.n_head
        self.dim = model.config.dim
        self.head_dim = self.dim // self.heads
        self.register_buffer("rotary", model.freqs_cis)

    def forward(self, tokens, conditioning, positions, *past):
        h = self.model.tok_embeddings_extended(tokens)
        condition = self.model.cls_embedding.cap_proj(conditioning)
        h = torch.where((positions == 0).reshape(1, -1, 1), condition, h)
        freqs = self.rotary[positions]
        presents = []
        for index, layer in enumerate(self.model.layers):
            x = layer.attention_norm(h)
            q, k, v = layer.attention.wqkv(x).chunk(3, dim=-1)
            shape = (1, tokens.shape[1], self.heads, self.head_dim)
            q = apply_rotary_emb(q.reshape(shape), freqs).transpose(1, 2)
            k = apply_rotary_emb(k.reshape(shape), freqs).transpose(1, 2)
            v = v.reshape(shape).transpose(1, 2)
            k = torch.cat((past[2 * index], k), dim=2)
            v = torch.cat((past[2 * index + 1], v), dim=2)
            # One masked dummy cache token avoids zero-sized WebGPU buffers.
            key_positions = torch.arange(k.shape[2], device=h.device) - 1
            mask = (key_positions[None, :] <= positions[:, None]) & (key_positions[None, :] >= 0)
            scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            probabilities = F.softmax(scores.masked_fill(~mask[None, None], -1e9), dim=-1)
            attention = (probabilities @ v).transpose(1, 2).reshape(1, -1, self.dim)
            h = h + layer.attention.wo(attention)
            h = h + layer.feed_forward(layer.ffn_norm(h))
            presents.extend((k, v))
        logits = self.model.output_extended(self.model.norm(h[:, -1:, :])).squeeze(1)
        return logits, *presents


class Decoder(nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.decoder = vq.decoder
        self.post_quant_conv = vq.post_quant_conv
        self.register_buffer("codebook", F.normalize(vq.quantize.embedding.weight, dim=-1))

    def forward(self, tokens):
        quant = self.codebook[tokens].reshape(1, 16, 16, 8).permute(0, 3, 1, 2)
        return self.decoder(self.post_quant_conv(quant)).mean(dim=1)


def export_variant(variant: str):
    model = GPT_models[f"GPT-{variant}"](model_type="ca_esm_embed_mean_pool").eval()
    state = torch.load(TOOLS / "checkpoints" / f"vermeer_{variant}.ckpt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    codes = np.load(TOOLS / "reference-codes.npy")[0].astype(np.int64)
    embedding = np.load(TOOLS / "reference-embedding.npy").astype(np.float32)
    prefix = torch.from_numpy(codes[:774]).reshape(1, -1)
    conditioning = torch.from_numpy(embedding).reshape(1, 1, -1)
    tokens = torch.cat((torch.zeros((1, 1), dtype=torch.int64), prefix), dim=1)
    positions = torch.arange(775)
    config = model.config
    past = tuple(torch.zeros(1, config.n_head, 1, config.dim // config.n_head) for _ in range(2 * config.n_layer))
    wrapper = BrowserGPT(model).eval()
    baseline = model(prefix, conditioning)[0][:, -1, :]
    logits, *present = wrapper(tokens, conditioning, positions, *past)
    torch.testing.assert_close(logits, baseline, atol=0.001, rtol=0.001)
    report = {"variant": variant, "pytorch_prefill_max_error": float((logits - baseline).abs().max())}
    inputs = ["tokens", "conditioning", "positions"] + [f"past_{i}" for i in range(len(past))]
    outputs = ["logits"] + [f"present_{i}" for i in range(len(past))]
    filename = f"vermeer-{variant.lower()}.onnx"
    report["onnx"] = export_checked(
        wrapper, (tokens, conditioning, positions, *past), filename,
        inputs, outputs,
        {"tokens": {1: "steps"}, "positions": {0: "steps"}} |
        {name: {2: "past_length"} for name in inputs[3:]} |
        {name: {2: "total_length"} for name in outputs[1:]},
        external_weights=True,
    )
    session = ort.InferenceSession(str(MODELS / filename))
    growing = prefix
    for offset in range(3):
        next_token = logits.argmax(-1).reshape(1, 1)
        growing = torch.cat((growing, next_token), dim=1)
        baseline = model(growing, conditioning)[0][:, -1, :]
        values = (next_token, conditioning, torch.tensor([775 + offset]), *present)
        logits, *present = wrapper(*values)
        torch.testing.assert_close(logits, baseline, atol=0.002, rtol=0.002)
        actual = session.run(None, dict(zip(inputs, [x.numpy() for x in values])))
        np.testing.assert_allclose(actual[0], logits.numpy(), atol=0.002, rtol=0.002)
    report["cached_steps_checked"] = 3
    save_json(OUTPUT / f"vermeer-{variant.lower()}-parity.json", report)
    del session, model, wrapper, state, past, present
    gc.collect()


def export_decoder():
    codes = np.load(TOOLS / "reference-codes.npy")[0].astype(np.int64)
    embedding = np.load(TOOLS / "reference-embedding.npy").astype(np.float32)
    report = {}
    vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).eval()
    state = torch.load(TOOLS / "checkpoints" / "vq_ds16_c2i.pt", map_location="cpu", weights_only=True)
    vq.load_state_dict(state["model"], strict=True)
    decoder = Decoder(vq).eval()
    channel_codes = torch.from_numpy(codes[1:257]).reshape(1, -1)
    report["decoder"] = export_checked(decoder, (channel_codes,), "decoder.onnx", ["tokens"], ["image"], atol=0.003)
    channels = []
    for channel in range(3):
        channel_tokens = torch.from_numpy(codes[channel * 258 + 1:channel * 258 + 257]).reshape(1, -1)
        channels.append(decoder(channel_tokens).numpy().reshape(256, 256))
    normalized_channels = [
        (channel - channel.min()) / max(float(channel.max() - channel.min()), 1e-8)
        for channel in channels
    ]
    nucleus, microtubules, er = normalized_channels
    rgb = np.stack((np.maximum(microtubules, er), er, nucleus), axis=-1)
    rgb = np.round(rgb * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb).save(OUTPUT / "reference.png")
    for name, channel in zip(["nucleus", "microtubules", "er"], channels):
        normalized = channel - channel.min()
        normalized = np.round(normalized / max(float(normalized.max()), 1e-8) * 255).astype(np.uint8)
        Image.fromarray(normalized).save(OUTPUT / f"landmark-{name}.png")
    save_json(OUTPUT / "reference.json", {
        "id": "A0A087WSY6_HPA047702_Rh30_fov_1",
        "cellLine": "Rh30", "prefix": codes[:774].tolist(),
        "publishedEmbedding": embedding.tolist(),
        "images": ["reference.png", "landmark-nucleus.png", "landmark-microtubules.png", "landmark-er.png"],
        "source": "https://github.com/microsoft/vermeer/tree/5c4ac7f1eeb722f1a8fe68c48743dc6a197806bf/example_data",
    })
    save_json(OUTPUT / "image-parity.json", report)


class ESMStart(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.embed_tokens

    def forward(self, tokens):
        return self.embedding(tokens)


class ESMLayer(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.layer = model.layers[index]
        self.rotary = model.rotary_emb

    def forward(self, hidden, positions):
        cos, sin = self.rotary(hidden, positions.reshape(1, -1))
        return self.layer(hidden, position_embeddings=(cos, sin))


class ESMEnd(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = model.norm

    def forward(self, hidden):
        # Match the supplied preparation script's two successive [1:-1] slices.
        return self.norm(hidden)[:, 2:-2, :].mean(dim=1).unsqueeze(1)


def export_esm():
    from transformers import EsmcModel
    revision = "fcc9d36f5fff97d9bb69a36e3252de61cc78c968"
    model = EsmcModel.from_pretrained(
        "biohub/ESMC-600M", revision=revision,
        attn_implementation="eager", dtype=torch.float32,
    ).eval()
    tokens = torch.tensor([[0, 20, 5, 6, 4, 7, 2]])
    hidden = model.embed_tokens(tokens)
    positions = torch.arange(tokens.shape[1])
    axes = {"hidden": {1: "length"}, "positions": {0: "length"}, "output": {1: "length"}}
    report = {"revision": revision, "layers": []}
    report["embedding"] = export_checked(
        ESMStart(model), (tokens,), "esm-start.onnx", ["tokens"], ["hidden"],
        {"tokens": {1: "length"}, "hidden": {1: "length"}},
    )
    for index in range(36):
        layer = ESMLayer(model, index)
        report["layers"].append(export_checked(
            layer, (hidden, positions), f"esm-layer-{index:02}.onnx",
            ["hidden", "positions"], ["output"], axes, atol=0.003,
        ))
        hidden = layer(hidden, positions)
    end = ESMEnd(model)
    report["pool"] = export_checked(
        end, (hidden,), "esm-end.onnx", ["hidden"], ["embedding"],
        {"hidden": {1: "length"}}, atol=0.003,
    )
    full = model(tokens).last_hidden_state[:, 2:-2].mean(1).unsqueeze(1)
    torch.testing.assert_close(end(hidden), full, atol=0.003, rtol=0.001)
    report["full_encoder_max_error"] = float((end(hidden) - full).abs().max())
    tokenizer = json.loads(Path(hf_hub_download("biohub/ESMC-600M", "tokenizer.json", revision=revision)).read_text())
    save_json(OUTPUT / "tokenizer.json", tokenizer["model"]["vocab"])
    save_json(OUTPUT / "esm-parity.json", report)


def manifest():
    required = ["decoder.onnx", "esm-start.onnx", "esm-end.onnx"]
    required += [f"esm-layer-{index:02}.onnx" for index in range(36)]
    if not all((MODELS / name).exists() for name in required):
        print("Exports incomplete; not publishing ready manifest.", flush=True)
        return
    files = {}
    variants = {}
    for variant, layers, heads, dim, params in [
        ("B", 12, 12, 768, 112), ("L", 24, 16, 1024, 344), ("XL", 36, 20, 1280, 777),
    ]:
        name = f"vermeer-{variant.lower()}.onnx"
        path = MODELS / name
        if not path.exists() or not (OUTPUT / f"vermeer-{variant.lower()}-parity.json").exists():
            continue
        graph = onnx.load(str(path), load_external_data=False)
        external = sorted({
            next(item.value for item in tensor.external_data if item.key == "location")
            for tensor in graph.graph.initializer if tensor.data_location == onnx.TensorProto.EXTERNAL
        })
        required.extend([name, *external])
        variants[variant] = {
            "name": f"Vermeer-{variant}", "graph": name, "externalData": external,
            "layers": layers, "heads": heads, "headDim": dim // heads, "paramsMillion": params,
        }
    for name in required:
        path = MODELS / name
        with path.open("rb") as stream:
            files[name] = {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    save_json(OUTPUT / "manifest.json", {
        "version": 2, "variants": variants, "precision": "FP32",
        "commonModels": ["decoder.onnx", "esm-start.onnx", "esm-end.onnx"] +
                        [f"esm-layer-{index:02}.onnx" for index in range(36)],
        "vermeerRevision": "5c4ac7f1eeb722f1a8fe68c48743dc6a197806bf",
        "esmRevision": "fcc9d36f5fff97d9bb69a36e3252de61cc78c968",
        "pooling": "legacy-double-trim", "files": files,
    })
    print("Published complete model manifest.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["images", "decoder", "esm", "manifest", "all"], default="all")
    parser.add_argument("--variants", nargs="+", choices=["B", "L", "XL"], default=["B", "L", "XL"])
    options = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    if options.stage in ("images", "all"):
        for variant in options.variants:
            export_variant(variant)
    if options.stage in ("images", "decoder", "all"):
        export_decoder()
    if options.stage in ("esm", "all"):
        export_esm()
    manifest()

"""Create independent PyTorch outputs for full-encoder and browser-decoder checks."""

import argparse
import json

import numpy as np
import torch
from transformers import EsmcModel

from export_models import Decoder, GPT_models, OUTPUT, TOOLS, VQ_models

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["B", "L", "XL"], default="B")
variant = parser.parse_args().variant
sequence = "MEAPAQLLFLLLLWLPDTTGEIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASIRATGIPARFSGSGSGTEFTLTISILQSEDFAVYYCQQYNNWP"
vocab = json.loads((OUTPUT / "tokenizer.json").read_text())
tokens = torch.tensor([[0, *[vocab[letter] for letter in sequence], 2]])
model = EsmcModel.from_pretrained(
    "biohub/ESMC-600M",
    revision="fcc9d36f5fff97d9bb69a36e3252de61cc78c968",
    attn_implementation="eager", dtype=torch.float32,
).eval()
embedding = model(tokens).last_hidden_state[:, 2:-2].mean(1).numpy().reshape(-1)
del model
gpt = GPT_models[f"GPT-{variant}"](model_type="ca_esm_embed_mean_pool").eval()
gpt.load_state_dict(
    torch.load(TOOLS / "checkpoints" / f"vermeer_{variant}.ckpt", map_location="cpu", weights_only=True),
    strict=True,
)
prefix = np.load(TOOLS / "reference-codes.npy")[0, :774].astype(np.int64)
conditioning = torch.from_numpy(embedding).reshape(1, 1, -1)
gpt_logits = []
gpt_tokens = []
growing = torch.from_numpy(prefix).reshape(1, -1)
for _ in range(4):
    logits = gpt(growing, conditioning)[0][:, -1, :]
    gpt_logits.append(logits.numpy().reshape(-1).tolist())
    token = int(logits.argmax(-1).item())
    gpt_tokens.append(token)
    growing = torch.cat((growing, torch.tensor([[token]])), dim=1)
del gpt
vq = VQ_models["VQ-16"]().eval()
vq.load_state_dict(
    torch.load(TOOLS / "checkpoints" / "vq_ds16_c2i.pt", map_location="cpu", weights_only=True)["model"],
    strict=True,
)
codes = np.load(TOOLS / "reference-codes.npy")[0, 1:257].astype(np.int64)
image = Decoder(vq)(torch.from_numpy(codes).reshape(1, 256)).numpy().reshape(-1)
(OUTPUT / f"browser-fixture-{variant.lower()}.json").write_text(json.dumps({
    "sequence": sequence,
    "embedding": embedding.tolist(),
    "decoderTokens": codes.tolist(),
    "decoderPixels": image.tolist(),
    "gptLogits": gpt_logits,
    "gptTokens": gpt_tokens,
}), encoding="utf-8")
print("Created independent full-encoder and decoder browser fixtures.")

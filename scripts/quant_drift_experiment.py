#!/usr/bin/env python
"""Phase 4 experiment: 1-bit weight quantization drift on the real Wan DiT.

Applies post-training 1-bit weight quantization (W -> sign(W) * scale, with a
per-output-channel scale = mean(|W|)) to the DiT's linear layers and measures
output drift over a short denoise loop, vs. the original model. Targets the FFN
first (38.5% of compute per Phase 1 Table 1), then all linear layers.

Weight-only (no activation quantization) for a clean measurement; the full
BitNet scheme also quantizes activations to 8-bit. Norms/embeddings are left in
float (standard BitNet practice). No model download (cached DiT; dummy text emb).
Quantization is post-training (no QAT) -- this measures the drift QAT must
later recover.

Usage:
    HF_TOKEN=... python scripts/quant_drift_experiment.py
"""
import sys

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
from diffusers import WanTransformer3DModel  # noqa: E402

STEPS = 4


@torch.no_grad()
def quantize_linear_(lin: nn.Linear) -> None:
    w = lin.weight.data
    scale = w.abs().float().mean(dim=1, keepdim=True).clamp(min=1e-8).to(w.dtype)
    lin.weight.data = w.sign() * scale


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    x_init = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    @torch.no_grad()
    def denoise():
        x = x_init.clone()
        tr = []
        for i in range(STEPS):
            t = torch.tensor([int((1 - i / STEPS) * 999)])
            v = model(x, t, txt, return_dict=False)[0]
            x = x - (1 / STEPS) * v
            tr.append(x.float().clone())
        return tr

    def cs(tr, o):
        return [F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item() for a, b in zip(tr, o)]

    def quantize(pred):
        n = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and pred(name):
                quantize_linear_(m)
                n += 1
        return n

    orig = denoise()
    n1 = quantize(lambda n: ".ffn." in n)
    print(f"FFN-only 1-bit ({n1} layers):  " + " ".join(f"{c:.3f}" for c in cs(denoise(), orig)))
    n2 = quantize(lambda n: ".ffn." not in n and "norm" not in n.lower())
    print(f"ALL-linear 1-bit (+{n2} layers): " + " ".join(f"{c:.3f}" for c in cs(denoise(), orig)))


if __name__ == "__main__":
    main()

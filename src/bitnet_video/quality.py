"""Quality evaluation for float vs quantized model outputs."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compare_layer_outputs(
    float_model: nn.Module,
    bit_model: nn.Module,
    input_tensor: torch.Tensor,
) -> List[Dict[str, object]]:
    """Compare intermediate outputs layer by layer.

    Hooks into both models to capture outputs at each named module,
    then computes per-layer quality metrics.

    Args:
        float_model: Original model.
        bit_model: Quantized model.
        input_tensor: Input tensor for forward pass.

    Returns:
        List of per-layer comparison metrics.
    """
    float_outputs = {}
    bit_outputs = {}

    def _make_hook(storage: dict, name: str):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            storage[name] = out.detach()
        return hook

    float_hooks = []
    bit_hooks = []

    for name, module in float_model.named_modules():
        if len(list(module.children())) == 0:
            float_hooks.append(module.register_forward_hook(_make_hook(float_outputs, name)))

    for name, module in bit_model.named_modules():
        if len(list(module.children())) == 0:
            bit_hooks.append(module.register_forward_hook(_make_hook(bit_outputs, name)))

    with torch.no_grad():
        try:
            float_model(input_tensor)
            bit_model(input_tensor)
        except Exception as e:
            logger.warning("Forward pass failed: %s", e)

    # Remove hooks
    for h in float_hooks + bit_hooks:
        h.remove()

    # Compare shared layers
    results = []
    for name in float_outputs:
        if name not in bit_outputs:
            continue
        fo = float_outputs[name].float()
        bo = bit_outputs[name].float()
        if fo.shape != bo.shape:
            continue

        mse = ((fo - bo) ** 2).mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            fo.flatten().unsqueeze(0), bo.flatten().unsqueeze(0)
        ).item()

        results.append({
            "name": name,
            "mse": round(mse, 8),
            "cosine_similarity": round(cosine, 6),
            "float_std": round(fo.std().item(), 6),
            "bit_std": round(bo.std().item(), 6),
            "has_nan": bool(torch.isnan(bo).any()),
        })

    return results


def compare_final_outputs(
    float_output: torch.Tensor,
    bit_output: torch.Tensor,
) -> Dict[str, float]:
    """Compare final model outputs.

    Args:
        float_output: Output from float model.
        bit_output: Output from quantized model.

    Returns:
        Quality metrics dictionary.
    """
    fo = float_output.float().flatten()
    bo = bit_output.float().flatten()

    mse = ((fo - bo) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        fo.unsqueeze(0), bo.unsqueeze(0)
    ).item()

    fo_norm = torch.norm(fo).item()
    diff_norm = torch.norm(fo - bo).item()
    relative_error = diff_norm / (fo_norm + 1e-8)

    signal_power = (fo ** 2).mean().item()
    noise_power = ((fo - bo) ** 2).mean().item()
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else float("inf")

    return {
        "mse": round(mse, 8),
        "cosine_similarity": round(cosine, 6),
        "relative_error": round(relative_error, 6),
        "snr_db": round(snr_db, 2),
        "float_mean": round(fo.mean().item(), 6),
        "bit_mean": round(bo.mean().item(), 6),
        "float_std": round(fo.std().item(), 6),
        "bit_std": round(bo.std().item(), 6),
        "num_nan": int(torch.isnan(bo).sum().item()),
    }


def analyze_weight_distribution(model: nn.Module) -> List[Dict[str, object]]:
    """Analyze weight distributions across all layers.

    Useful for understanding how well weights will quantize to 1-bit.
    Layers with balanced positive/negative weights quantize better.

    Args:
        model: Model to analyze.

    Returns:
        Per-layer weight statistics.
    """
    results = []

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        w = param.float()
        pos_frac = (w > 0).float().mean().item()
        neg_frac = (w < 0).float().mean().item()
        zero_frac = (w == 0).float().mean().item()

        # How much information is lost by binarization
        w_binary = w.sign()
        w_binary[w_binary == 0] = 1.0
        alpha = w.abs().mean()
        reconstructed = w_binary * alpha
        quant_mse = ((w - reconstructed) ** 2).mean().item()
        orig_var = w.var().item() + 1e-8

        results.append({
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "mean": round(w.mean().item(), 6),
            "std": round(w.std().item(), 6),
            "min": round(w.min().item(), 6),
            "max": round(w.max().item(), 6),
            "pos_fraction": round(pos_frac, 4),
            "neg_fraction": round(neg_frac, 4),
            "zero_fraction": round(zero_frac, 6),
            "balance": round(min(pos_frac, neg_frac) / max(pos_frac, neg_frac, 1e-8), 4),
            "quant_relative_mse": round(quant_mse / orig_var, 6),
        })

    return results

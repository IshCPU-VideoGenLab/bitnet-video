"""Convert float models to 1-bit quantized models.

Walks through a model's modules and replaces nn.Linear with BitLinear
and nn.Conv2d with BitConv2d, skipping sensitive layers like normalization.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from bitnet_video.config import QuantConfig
from bitnet_video.bitlinear import BitLinear
from bitnet_video.bitconv import BitConv2d

logger = logging.getLogger(__name__)


@dataclass
class ConversionReport:
    """Report of model quantization conversion.

    Args:
        total_modules: Total modules inspected.
        quantized_linear: Number of Linear → BitLinear conversions.
        quantized_conv: Number of Conv2d → BitConv2d conversions.
        skipped: Number of modules skipped (kept in float).
        skipped_names: Names of skipped modules.
        original_size_mb: Original model size in MB.
        quantized_size_mb: Quantized model size in MB (estimated).
        original_params: Total original parameters.
        quantized_params: Total quantized parameters.
    """

    total_modules: int = 0
    quantized_linear: int = 0
    quantized_conv: int = 0
    skipped: int = 0
    skipped_names: List[str] = field(default_factory=list)
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    original_params: int = 0
    quantized_params: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.quantized_size_mb == 0:
            return 0.0
        return self.original_size_mb / self.quantized_size_mb

    @property
    def total_quantized(self) -> int:
        return self.quantized_linear + self.quantized_conv


def _estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB from parameter dtypes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def _estimate_1bit_size_mb(model: nn.Module, config: QuantConfig) -> float:
    """Estimate model size if quantizable layers were 1-bit."""
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, BitLinear)):
            if config.should_skip(name):
                total_bits += sum(p.numel() * p.element_size() * 8 for p in module.parameters())
            else:
                # 1-bit weights + scale factors
                total_bits += sum(p.numel() for p in module.parameters())
                total_bits += 32  # Scale factor overhead
        elif isinstance(module, (nn.Conv2d, BitConv2d)):
            if config.should_skip(name):
                total_bits += sum(p.numel() * p.element_size() * 8 for p in module.parameters())
            else:
                total_bits += sum(p.numel() for p in module.parameters())
                total_bits += 32
        # Other modules keep their original size (not double-counted through children)
    return total_bits / 8 / (1024 ** 2)


def _replace_module_at_path(
    model: nn.Module,
    path: str,
    new_module: nn.Module,
) -> None:
    """Replace a module at a dotted path in the model."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = new_module
    else:
        setattr(parent, final, new_module)


def quantize_model(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
) -> tuple:
    """Convert a float model to 1-bit quantized model.

    Replaces nn.Linear → BitLinear and nn.Conv2d → BitConv2d for all
    layers that don't match skip patterns.

    Args:
        model: The model to quantize (modified in-place).
        config: Quantization configuration.

    Returns:
        Tuple of (quantized_model, ConversionReport).
    """
    if config is None:
        config = QuantConfig()

    report = ConversionReport()
    report.original_params = sum(p.numel() for p in model.parameters())
    report.original_size_mb = _estimate_model_size_mb(model)

    # Collect replacements (can't modify during iteration)
    replacements = []

    for name, module in model.named_modules():
        report.total_modules += 1

        if config.should_skip(name):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                report.skipped += 1
                report.skipped_names.append(name)
                logger.debug("Skipping (pattern match): %s", name)
            continue

        if isinstance(module, nn.Linear) and config.quantize_linear:
            replacements.append((name, "linear", module))
        elif isinstance(module, nn.Conv2d) and config.quantize_conv:
            replacements.append((name, "conv", module))

    # Apply replacements
    for name, layer_type, module in replacements:
        if layer_type == "linear":
            new_module = BitLinear.from_linear(module, config.activation_bits)
            _replace_module_at_path(model, name, new_module)
            report.quantized_linear += 1
            logger.debug("Quantized Linear: %s", name)
        elif layer_type == "conv":
            new_module = BitConv2d.from_conv2d(module, config.activation_bits)
            _replace_module_at_path(model, name, new_module)
            report.quantized_conv += 1
            logger.debug("Quantized Conv2d: %s", name)

    report.quantized_params = sum(p.numel() for p in model.parameters())
    report.quantized_size_mb = round(_estimate_model_size_mb(model), 2)

    gc.collect()

    logger.info(
        "Quantization complete: %d Linear + %d Conv2d quantized, %d skipped. "
        "Size: %.1f MB → %.1f MB (estimated 1-bit: %.1f MB)",
        report.quantized_linear, report.quantized_conv, report.skipped,
        report.original_size_mb, report.quantized_size_mb,
        report.original_size_mb / 16,
    )

    return model, report

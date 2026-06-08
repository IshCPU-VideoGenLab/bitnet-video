"""Configuration for bitnet-video quantization."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class QuantConfig:
    """Configuration for model quantization.

    Args:
        weight_bits: Number of bits for weights (1 for BitNet).
        activation_bits: Number of bits for activations (8 typical).
        skip_patterns: Module name patterns to skip (keep in float).
        quantize_linear: Whether to quantize nn.Linear layers.
        quantize_conv: Whether to quantize nn.Conv2d layers.
        symmetric: Use symmetric quantization for activations.
        per_channel: Per-channel weight scaling (vs per-tensor).
        output_dir: Directory for results.
        model_name: Model being quantized.
        model_path: Local path to model weights.
        dtype: Original model dtype.
        verbose: Print progress.
    """

    weight_bits: int = 1
    activation_bits: int = 8
    skip_patterns: List[str] = field(default_factory=lambda: [
        "norm", "embed", "ln", "layernorm", "rmsnorm",
        "pos_embed", "patch_embed", "cls_token",
    ])
    quantize_linear: bool = True
    quantize_conv: bool = True
    symmetric: bool = True
    per_channel: bool = False
    output_dir: str = "results"
    model_name: str = "wan-1.3b"
    model_path: Optional[str] = None
    dtype: str = "float16"
    verbose: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.weight_bits not in (1, 2, 4, 8):
            raise ValueError(f"weight_bits must be 1, 2, 4, or 8, got {self.weight_bits}")
        if self.activation_bits not in (4, 8, 16, 32):
            raise ValueError(f"activation_bits must be 4, 8, 16, or 32, got {self.activation_bits}")

    def should_skip(self, module_name: str) -> bool:
        """Check if a module should be skipped (kept in float).

        Args:
            module_name: Fully qualified module name.

        Returns:
            True if the module matches any skip pattern.
        """
        name_lower = module_name.lower()
        return any(pattern.lower() in name_lower for pattern in self.skip_patterns)

"""1-bit Linear layer (BitLinear) — drop-in replacement for nn.Linear.

Implements the BitNet linear layer where weights are binarized to {-1, +1}
and activations are quantized to 8-bit during the forward pass.

Uses Straight-Through Estimator (STE) for gradient computation through
the sign function during training.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_video.quantize import weight_quant_1bit, activation_quant_absmax

logger = logging.getLogger(__name__)


class STESign(torch.autograd.Function):
    """Sign function with Straight-Through Estimator for gradients.

    Forward: y = sign(x)
    Backward: dy/dx = 1 (identity, straight-through)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        result = x.sign()
        result[result == 0] = 1.0
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # Straight-through


class BitLinear(nn.Module):
    """1-bit Linear layer.

    Stores weights in full precision but binarizes them during forward pass.
    This is "quantization-aware" behavior — the model learns weights that
    quantize well.

    Forward pass:
        1. Binarize weights: W_bin = sign(W), α = mean(|W|)
        2. Quantize activations: x_q = absmax_quant(x, 8 bits)
        3. Compute: y = x_q @ W_bin.T × α × β
        4. Optional bias

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        activation_bits: Bits for activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits

        # Full-precision weight (binarized during forward)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMSNorm before quantization (as in BitNet b1.58)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values favorable for binarization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale down to encourage balanced +1/-1 distribution
        with torch.no_grad():
            self.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 1-bit weights and quantized activations.

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        # Normalize input
        x_norm = self.norm(x)

        # Quantize activations to N bits
        x_quant, x_scale = activation_quant_absmax(x_norm, self.activation_bits)

        # Binarize weights with STE for training
        if self.training:
            w_binary = STESign.apply(self.weight)
        else:
            w_binary = self.weight.sign()
            w_binary[w_binary == 0] = 1.0

        # Weight scale factor
        w_scale = self.weight.float().abs().mean().clamp(min=1e-8).to(x.dtype)

        # Linear operation with binarized weights
        # In a real BitNet kernel, this would be XNOR + popcount
        # Here we simulate with standard matmul on {-1, +1} values
        y = F.linear(x_quant, w_binary)

        # Apply scaling
        y = y * w_scale

        # Bias
        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, activation_bits: int = 8) -> "BitLinear":
        """Create a BitLinear from an existing nn.Linear, copying weights.

        Args:
            linear: Source nn.Linear layer.
            activation_bits: Bits for activation quantization.

        Returns:
            A new BitLinear with copied weights.
        """
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            activation_bits=activation_bits,
        )

        with torch.no_grad():
            bit_linear.weight.copy_(linear.weight)
            if linear.bias is not None and bit_linear.bias is not None:
                bit_linear.bias.copy_(linear.bias)

        return bit_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation_bits={self.activation_bits}"
        )

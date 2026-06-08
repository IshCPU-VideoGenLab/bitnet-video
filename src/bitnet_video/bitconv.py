"""1-bit Conv2d layer — drop-in replacement for nn.Conv2d.

Same quantization scheme as BitLinear but applied to convolution weights.
"""

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_video.quantize import activation_quant_absmax
from bitnet_video.bitlinear import STESign

logger = logging.getLogger(__name__)


class BitConv2d(nn.Module):
    """1-bit Conv2d layer.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        groups: Convolution groups.
        bias: Whether to include bias.
        activation_bits: Bits for activation quantization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_bits = activation_bits

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 1-bit weights.

        Args:
            x: Input tensor, shape (batch, in_channels, H, W).

        Returns:
            Output tensor.
        """
        # Quantize activations
        x_quant, _ = activation_quant_absmax(x, self.activation_bits)

        # Binarize weights
        if self.training:
            w_binary = STESign.apply(self.weight)
        else:
            w_binary = self.weight.sign()
            w_binary[w_binary == 0] = 1.0

        w_scale = self.weight.float().abs().mean().clamp(min=1e-8).to(x.dtype)

        # Convolution with binary weights
        y = F.conv2d(x_quant, w_binary, None, self.stride, self.padding, groups=self.groups)
        y = y * w_scale

        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)

        return y

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, activation_bits: int = 8) -> "BitConv2d":
        """Create from existing nn.Conv2d, copying weights.

        Args:
            conv: Source Conv2d layer.
            activation_bits: Activation quantization bits.

        Returns:
            New BitConv2d with copied weights.
        """
        bit_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=conv.bias is not None,
            activation_bits=activation_bits,
        )
        with torch.no_grad():
            bit_conv.weight.copy_(conv.weight)
            if conv.bias is not None and bit_conv.bias is not None:
                bit_conv.bias.copy_(conv.bias)
        return bit_conv

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, "
            f"activation_bits={self.activation_bits}"
        )

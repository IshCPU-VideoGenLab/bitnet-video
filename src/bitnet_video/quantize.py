"""Core quantization functions for 1-bit weights and low-bit activations.

Implements the BitNet quantization scheme:
- Weights: sign(W) × α, where α = mean(|W|)
- Activations: symmetric absmax quantization to N bits

Reference: https://arxiv.org/abs/2310.11453
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def weight_quant_1bit(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to 1-bit using sign function with mean absolute scaling.

    W_quant = sign(W) × α, where α = mean(|W|)

    The sign function maps: negative → -1, zero → +1, positive → +1

    Args:
        weight: Float weight tensor of any shape.

    Returns:
        Tuple of (quantized_weight, scale_factor).
        quantized_weight has values in {-1, +1} stored as float.
        scale_factor α is a scalar tensor.
    """
    # Compute scale in float32 for numerical stability
    w_float = weight.float()
    alpha = w_float.abs().mean()

    # Clamp alpha to avoid division by zero
    alpha = alpha.clamp(min=1e-8)

    # Sign quantization: sign(0) = +1 by convention
    w_binary = w_float.sign()
    w_binary[w_binary == 0] = 1.0

    # Return in original dtype
    return w_binary.to(weight.dtype), alpha.to(weight.dtype)


def weight_quant_1bit_per_channel(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 1-bit quantization (separate scale per output channel).

    Args:
        weight: Float weight tensor, shape (out_features, in_features, ...).

    Returns:
        Tuple of (quantized_weight, scale_per_channel).
        scale_per_channel has shape (out_features,).
    """
    w_float = weight.float()

    # Flatten all dims except first (output channels)
    flat = w_float.reshape(w_float.shape[0], -1)
    alpha = flat.abs().mean(dim=1).clamp(min=1e-8)  # (out_features,)

    w_binary = w_float.sign()
    w_binary[w_binary == 0] = 1.0

    # Reshape alpha for broadcasting
    shape = [alpha.shape[0]] + [1] * (weight.dim() - 1)
    alpha = alpha.reshape(shape)

    return w_binary.to(weight.dtype), alpha.to(weight.dtype)


def activation_quant_absmax(
    x: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax activation quantization.

    Maps activations to [-Q_max, +Q_max] where Q_max = 2^(bits-1) - 1.

    Args:
        x: Activation tensor.
        bits: Number of quantization bits.

    Returns:
        Tuple of (quantized_x, scale).
    """
    q_max = (1 << (bits - 1)) - 1  # e.g., 127 for 8-bit

    # Compute scale from max absolute value
    x_float = x.float()
    x_abs_max = x_float.abs().max().clamp(min=1e-8)
    scale = x_abs_max / q_max

    # Quantize
    x_quant = (x_float / scale).round().clamp(-q_max, q_max)

    # Dequantize back to float (simulated quantization)
    x_deq = x_quant * scale

    return x_deq.to(x.dtype), scale.to(x.dtype)


def activation_quant_per_token(
    x: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token activation quantization (separate scale per sequence position).

    Args:
        x: Activation tensor, shape (..., features).
        bits: Number of quantization bits.

    Returns:
        Tuple of (quantized_x, scale_per_token).
    """
    q_max = (1 << (bits - 1)) - 1

    x_float = x.float()

    # Scale per token (all dims except last)
    flat = x_float.reshape(-1, x_float.shape[-1])
    token_max = flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scale = token_max / q_max  # (num_tokens, 1)

    x_quant = (flat / scale).round().clamp(-q_max, q_max)
    x_deq = (x_quant * scale).reshape(x.shape)

    return x_deq.to(x.dtype), scale.squeeze().to(x.dtype)


def pack_binary_weights(weight_binary: torch.Tensor) -> torch.Tensor:
    """Pack 1-bit weights into int8 tensors for storage efficiency.

    Each int8 stores 8 binary weights. Reduces memory 8× compared
    to storing as float.

    Args:
        weight_binary: Binary weight tensor with values in {-1, +1}.

    Returns:
        Packed int8 tensor. Shape has last dim divided by 8.
    """
    # Map {-1, +1} → {0, 1}
    bits = ((weight_binary.float().sign() + 1) / 2).to(torch.uint8)

    # Flatten for packing
    flat = bits.reshape(-1)
    # Pad to multiple of 8
    pad_len = (8 - flat.shape[0] % 8) % 8
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8)])

    # Pack 8 bits per byte
    flat = flat.reshape(-1, 8)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8)
    for i in range(8):
        packed |= (flat[:, i] << (7 - i))

    return packed


def unpack_binary_weights(
    packed: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Unpack int8-packed binary weights back to {-1, +1} float tensor.

    Args:
        packed: Packed int8 tensor from pack_binary_weights().
        original_shape: Original weight tensor shape.

    Returns:
        Unpacked float tensor with values in {-1, +1}.
    """
    total_elements = 1
    for s in original_shape:
        total_elements *= s

    # Unpack bits
    bits = []
    for i in range(8):
        bits.append((packed >> (7 - i)) & 1)
    unpacked = torch.stack(bits, dim=-1).reshape(-1)[:total_elements]

    # Map {0, 1} → {-1, +1}
    result = unpacked.float() * 2 - 1
    return result.reshape(original_shape)


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    """Compute the relative quantization error.

    Args:
        original: Original float weight.
        quantized: Quantized weight (binary values).
        scale: Scaling factor.

    Returns:
        Relative MSE (quantization error / original magnitude).
    """
    reconstructed = quantized.float() * scale.float()
    mse = ((original.float() - reconstructed) ** 2).mean().item()
    orig_var = original.float().var().item() + 1e-8
    return mse / orig_var

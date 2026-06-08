"""Tests for bitnet_video.quantize module."""

import pytest
import torch

from bitnet_video.quantize import (
    weight_quant_1bit,
    weight_quant_1bit_per_channel,
    activation_quant_absmax,
    pack_binary_weights,
    unpack_binary_weights,
    compute_quantization_error,
)


class TestWeightQuant1Bit:
    def test_output_values(self) -> None:
        w = torch.randn(64, 32)
        w_q, alpha = weight_quant_1bit(w)
        unique = w_q.unique().tolist()
        assert set(unique).issubset({-1.0, 1.0})

    def test_scale_positive(self) -> None:
        w = torch.randn(32, 32)
        _, alpha = weight_quant_1bit(w)
        assert alpha.item() > 0

    def test_zero_weights(self) -> None:
        w = torch.zeros(16, 16)
        w_q, alpha = weight_quant_1bit(w)
        assert (w_q == 1.0).all()

    def test_shape_preserved(self) -> None:
        w = torch.randn(128, 64)
        w_q, _ = weight_quant_1bit(w)
        assert w_q.shape == w.shape

    def test_float16_input(self) -> None:
        w = torch.randn(32, 32, dtype=torch.float16)
        w_q, alpha = weight_quant_1bit(w)
        assert w_q.dtype == torch.float16


class TestPerChannelQuant:
    def test_per_channel_shapes(self) -> None:
        w = torch.randn(64, 32)
        w_q, alpha = weight_quant_1bit_per_channel(w)
        assert w_q.shape == w.shape
        assert alpha.shape[0] == 64


class TestActivationQuant:
    def test_output_shape(self) -> None:
        x = torch.randn(4, 16, 64)
        x_q, scale = activation_quant_absmax(x, bits=8)
        assert x_q.shape == x.shape

    def test_range_bounded(self) -> None:
        x = torch.randn(4, 64)
        x_q, scale = activation_quant_absmax(x, bits=8)
        reconstructed = x_q  # Already dequantized
        assert reconstructed.abs().max() <= x.abs().max() * 1.1


class TestPacking:
    def test_pack_unpack_roundtrip(self) -> None:
        w = torch.randn(64, 32)
        w_binary = w.sign()
        w_binary[w_binary == 0] = 1.0
        packed = pack_binary_weights(w_binary)
        unpacked = unpack_binary_weights(packed, w_binary.shape)
        assert torch.allclose(w_binary, unpacked)

    def test_packed_smaller(self) -> None:
        w_binary = torch.ones(1024)
        packed = pack_binary_weights(w_binary)
        assert packed.numel() < w_binary.numel()


class TestQuantError:
    def test_error_finite(self) -> None:
        w = torch.randn(64, 32)
        w_q, alpha = weight_quant_1bit(w)
        error = compute_quantization_error(w, w_q, alpha)
        assert error >= 0
        assert not torch.isnan(torch.tensor(error))

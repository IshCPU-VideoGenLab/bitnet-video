"""Tests for bitnet_video.bitlinear module."""

import pytest
import torch
import torch.nn as nn

from bitnet_video.bitlinear import BitLinear


class TestBitLinearShape:
    def test_output_shape(self) -> None:
        layer = BitLinear(64, 128)
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 16, 128)

    def test_no_bias(self) -> None:
        layer = BitLinear(32, 64, bias=False)
        assert layer.bias is None
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 8, 64)


class TestBitLinearGradient:
    def test_gradient_flows(self) -> None:
        layer = BitLinear(32, 32)
        layer.train()
        x = torch.randn(1, 4, 32, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    def test_weight_gets_gradient(self) -> None:
        layer = BitLinear(32, 32)
        layer.train()
        x = torch.randn(1, 4, 32)
        y = layer(x)
        y.sum().backward()
        assert layer.weight.grad is not None


class TestBitLinearNumerical:
    def test_no_nan(self) -> None:
        layer = BitLinear(64, 64)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            y = layer(x)
        assert not torch.isnan(y).any()

    def test_no_inf(self) -> None:
        layer = BitLinear(64, 64)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            y = layer(x)
        assert not torch.isinf(y).any()


class TestBitLinearFromLinear:
    def test_conversion(self) -> None:
        linear = nn.Linear(64, 128)
        bit = BitLinear.from_linear(linear)
        assert bit.in_features == 64
        assert bit.out_features == 128
        assert torch.allclose(bit.weight.data, linear.weight.data)

    def test_converted_forward(self) -> None:
        linear = nn.Linear(32, 32)
        bit = BitLinear.from_linear(linear)
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            y = bit(x)
        assert y.shape == (1, 4, 32)

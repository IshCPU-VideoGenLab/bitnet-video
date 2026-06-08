"""Tests for bitnet_video.converter module."""

import pytest
import torch
import torch.nn as nn

from bitnet_video.config import QuantConfig
from bitnet_video.bitlinear import BitLinear
from bitnet_video.bitconv import BitConv2d
from bitnet_video.converter import quantize_model


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(32, 64)
        self.layer1 = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.layer2 = nn.Linear(64, 32)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.norm(self.layer1(self.embed(x))))


class TestQuantizeModel:
    def test_linear_converted(self) -> None:
        model = DummyModel()
        config = QuantConfig(skip_patterns=["norm", "embed"])
        model, report = quantize_model(model, config)
        assert isinstance(model.layer1, BitLinear)
        assert isinstance(model.layer2, BitLinear)

    def test_skip_patterns(self) -> None:
        model = DummyModel()
        config = QuantConfig(skip_patterns=["norm", "embed"])
        model, report = quantize_model(model, config)
        # embed has "embed" in name, norm has "norm" — both skipped
        assert isinstance(model.embed, nn.Linear)
        assert isinstance(model.norm, nn.LayerNorm)

    def test_conv_converted(self) -> None:
        model = DummyModel()
        config = QuantConfig(skip_patterns=["norm", "embed"], quantize_conv=True)
        model, report = quantize_model(model, config)
        assert isinstance(model.conv, BitConv2d)

    def test_report_counts(self) -> None:
        model = DummyModel()
        config = QuantConfig(skip_patterns=["norm", "embed"])
        _, report = quantize_model(model, config)
        assert report.quantized_linear == 2  # layer1, layer2
        assert report.skipped >= 1  # embed

    def test_forward_after_quantize(self) -> None:
        model = DummyModel()
        config = QuantConfig(skip_patterns=["norm", "embed"])
        model, _ = quantize_model(model, config)
        model.eval()
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 8, 32)
        assert not torch.isnan(y).any()

    def test_no_quantize(self) -> None:
        model = DummyModel()
        config = QuantConfig(quantize_linear=False, quantize_conv=False)
        model, report = quantize_model(model, config)
        assert report.total_quantized == 0

"""Tests for bitnet_video.benchmark module."""

import pytest

from bitnet_video.benchmark import LayerBenchmark, benchmark_linear_layers


class TestLayerBenchmark:
    def test_speedup(self) -> None:
        r = LayerBenchmark(label="test", in_features=64, out_features=64,
                          float_time_ms=10.0, bit_time_ms=5.0)
        assert r.speedup == pytest.approx(2.0)

    def test_zero_bit_time(self) -> None:
        r = LayerBenchmark(label="test", in_features=64, out_features=64,
                          float_time_ms=10.0, bit_time_ms=0.0)
        assert r.speedup == 0.0


class TestBenchmarkLinear:
    def test_runs(self) -> None:
        results = benchmark_linear_layers(
            sizes=[(64, 64)], num_warmup=1, num_steps=2, batch_size=1, seq_len=8,
        )
        assert len(results) == 1
        assert results[0].float_time_ms > 0
        assert results[0].bit_time_ms > 0

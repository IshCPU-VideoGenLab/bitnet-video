"""Speed and memory benchmarking for float vs 1-bit models."""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerBenchmark:
    """Benchmark result for a single layer type/size."""
    label: str
    in_features: int
    out_features: int
    float_time_ms: float = 0.0
    bit_time_ms: float = 0.0
    float_memory_mb: float = 0.0
    bit_memory_mb: float = 0.0

    @property
    def speedup(self) -> float:
        if self.bit_time_ms == 0:
            return 0.0
        return self.float_time_ms / self.bit_time_ms


def benchmark_linear_layers(
    sizes: List[Tuple[int, int]] = None,
    num_warmup: int = 3,
    num_steps: int = 10,
    batch_size: int = 4,
    seq_len: int = 64,
) -> List[LayerBenchmark]:
    """Benchmark nn.Linear vs BitLinear at various sizes.

    Args:
        sizes: List of (in_features, out_features) to test.
        num_warmup: Warmup iterations.
        num_steps: Measurement iterations.
        batch_size: Batch size for input.
        seq_len: Sequence length for input.

    Returns:
        List of LayerBenchmark results.
    """
    from bitnet_video.bitlinear import BitLinear

    if sizes is None:
        sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    results = []

    for in_f, out_f in sizes:
        logger.info("Benchmarking Linear %d → %d", in_f, out_f)

        # Float layer
        float_layer = nn.Linear(in_f, out_f, bias=False).float()
        float_layer.eval()
        x_float = torch.randn(batch_size, seq_len, in_f)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                float_layer(x_float)

        # Time float
        float_times = []
        with torch.no_grad():
            for _ in range(num_steps):
                gc.collect()
                start = time.perf_counter()
                float_layer(x_float)
                float_times.append((time.perf_counter() - start) * 1000)

        # Bit layer
        bit_layer = BitLinear.from_linear(float_layer)
        bit_layer.eval()

        with torch.no_grad():
            for _ in range(num_warmup):
                bit_layer(x_float)

        bit_times = []
        with torch.no_grad():
            for _ in range(num_steps):
                gc.collect()
                start = time.perf_counter()
                bit_layer(x_float)
                bit_times.append((time.perf_counter() - start) * 1000)

        # Memory
        float_mem = sum(p.numel() * p.element_size() for p in float_layer.parameters()) / (1024**2)
        bit_mem = sum(p.numel() * p.element_size() for p in bit_layer.parameters()) / (1024**2)

        result = LayerBenchmark(
            label=f"Linear({in_f}→{out_f})",
            in_features=in_f,
            out_features=out_f,
            float_time_ms=sum(float_times) / len(float_times),
            bit_time_ms=sum(bit_times) / len(bit_times),
            float_memory_mb=float_mem,
            bit_memory_mb=bit_mem,
        )
        results.append(result)

        logger.info(
            "  Float: %.2f ms, Bit: %.2f ms, Speedup: %.2fx",
            result.float_time_ms, result.bit_time_ms, result.speedup,
        )

    return results


def benchmark_model_inference(
    float_model: nn.Module,
    bit_model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 2,
    num_steps: int = 5,
) -> Dict[str, float]:
    """Benchmark full model inference: float vs quantized.

    Args:
        float_model: Original float model.
        bit_model: Quantized model.
        input_tensor: Input for forward pass.
        num_warmup: Warmup iterations.
        num_steps: Measurement iterations.

    Returns:
        Dictionary with timing and speedup.
    """
    float_model.eval()
    bit_model.eval()

    def _time_model(model: nn.Module, x: torch.Tensor) -> float:
        with torch.no_grad():
            for _ in range(num_warmup):
                try:
                    model(x)
                except Exception:
                    model(x.reshape(x.shape[0], -1)[:, :64])
        times = []
        with torch.no_grad():
            for _ in range(num_steps):
                gc.collect()
                start = time.perf_counter()
                try:
                    model(x)
                except Exception:
                    model(x.reshape(x.shape[0], -1)[:, :64])
                times.append((time.perf_counter() - start) * 1000)
        return sum(times) / len(times)

    float_time = _time_model(float_model, input_tensor)
    bit_time = _time_model(bit_model, input_tensor)

    return {
        "float_time_ms": round(float_time, 3),
        "bit_time_ms": round(bit_time, 3),
        "speedup": round(float_time / bit_time, 3) if bit_time > 0 else 0,
        "float_params": sum(p.numel() for p in float_model.parameters()),
        "bit_params": sum(p.numel() for p in bit_model.parameters()),
    }

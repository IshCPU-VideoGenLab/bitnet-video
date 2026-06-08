<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# bitnet-video

**1-bit weight quantization for CPU-native video generation — turning matrix multiply into XNOR + popcount.**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab) — building the first video generation model that trains and runs entirely on commodity CPUs.

---

## Why 1-Bit?

GPUs dominate AI because of fast floating-point matrix multiplication. But when weights are constrained to {-1, +1}, multiplication becomes XNOR and accumulation becomes popcount — operations CPUs execute at billions per second through their integer ALUs. The GPU's advantage disappears.

**bitnet-video** applies BitNet-style 1-bit quantization to video generation models, reducing memory 16× and enabling binary computation on commodity CPUs.

---

## The Bigger Picture

**Phase 4** of a 7-phase research project:

| Phase | Repo | Speedup |
|-------|------|---------|
| 1 | [wan-profiler](https://github.com/IshCPU-VideoGenLab/wan-profiler) | (profiling) |
| 2 | [mamba-video](https://github.com/IshCPU-VideoGenLab/mamba-video) | ~2× (O(n²)→O(n)) |
| 3 | [codec-video-gen](https://github.com/IshCPU-VideoGenLab/codec-video-gen) | ~4-6× (temporal) |
| **4** | **bitnet-video** (this repo) | **~8-16× (binary ops)** |
| 5 | simd-kernels | Hardware-native (AVX2 + NEON) |
| 6 | (distributed) | Multi-machine |
| 7 | cpu-video-gen | Full pipeline |

Combined theoretical speedup: **64-192×**

---

## Features

- **BitLinear** — drop-in replacement for `nn.Linear` with 1-bit weights
- **BitConv2d** — drop-in replacement for `nn.Conv2d` with 1-bit weights
- **Model converter** — automatically swap float layers for 1-bit versions
- **Mixed precision** — keep sensitive layers (normalization, embeddings) in float
- **Weight analysis** — visualize weight distributions pre/post quantization
- **Speed benchmarks** — compare float vs binary matmul on CPU

---

## Installation

```bash
git clone https://github.com/IshCPU-VideoGenLab/bitnet-video.git
cd bitnet-video
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

---

## Usage

```bash
# Quantize model to 1-bit
python scripts/run_quantize.py --model wan-1.3b --output results/

# Benchmark float vs quantized
python scripts/run_benchmark.py --original wan-1.3b --quantized results/model_1bit/

# Analyze weight distributions
python scripts/analyze_weights.py --model wan-1.3b --output results/
```

### Python API

```python
from bitnet_video.bitlinear import BitLinear
from bitnet_video.converter import quantize_model

# Single layer replacement
bit_layer = BitLinear(in_features=512, out_features=512)

# Full model conversion
quantized_model = quantize_model(original_model, skip_patterns=["norm", "embed"])
```

---

## How 1-Bit Quantization Works

### Weight Binarization

```
Float weight:     [-0.3,  0.7, -0.1,  0.5, -0.8,  0.2]
Sign function:    [-1,    +1,   -1,   +1,   -1,   +1  ]
Packed bits:       0      1     0     1     0     1     → 0b010101
```

The sign function quantizes each weight to {-1, +1}. A scaling factor α (mean absolute value) preserves the magnitude:

```
W_binary = sign(W) × α,  where α = mean(|W|)
```

### Binary Matrix Multiplication

Standard: `y = x @ W` (float multiply-accumulate)

Binary: `y = popcount(xnor(x_bits, W_bits)) × α × β`

Where β is the activation scaling factor. XNOR + popcount replaces all float arithmetic.

### What Stays in Float

- **Layer normalization** — needs precise statistics
- **Embeddings** — lookup tables, not matmul
- **Activation functions** — element-wise, cheap anyway
- **Final output projection** — quality-sensitive

---

## Citation

```bibtex
@software{kwakye2026bitnetvideo,
  author = {Kwakye, Ishmael Affum},
  title = {bitnet-video: 1-Bit Quantization for CPU-Native Video Generation},
  year = {2026},
  url = {https://github.com/IshCPU-VideoGenLab/bitnet-video},
  institution = {University of Ghana, Legon}
}
```

## References

- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — Wang et al., 2023
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) — Ma et al., 2024

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Phase 4 of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab). Eliminating the GPU's last advantage — floating-point arithmetic.*

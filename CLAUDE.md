# CLAUDE.md — bitnet-video

> Read by Claude Code at the start of every session.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** bitnet-video
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon
- **Phase:** 4 of 7

---

## What This Project Is

This is the **quantization** phase. We apply BitNet-style 1-bit quantization
to the modified video generation model (post Phase 2 Mamba surgery + Phase 3
codec design). The goal: replace floating-point matrix multiplication with
binary operations (XNOR + popcount) that CPUs execute billions of times per second.

**bitnet-video** implements:

1. **1-bit weight quantization** — weights constrained to {-1, +1}
2. **Low-bit activation quantization** — activations quantized to 8-bit or lower
3. **Quantization-aware modules** — drop-in replacements for nn.Linear and Conv2d
4. **Quality measurement** — how much does 1-bit cost in output quality?
5. **Speed benchmarking** — how much faster are binary ops on CPU?
6. **Integration** — apply to both the full model (I-frames) and delta predictor (P-frames)

---

## Why 1-Bit Quantization?

### The GPU Advantage Is Float Matmul

GPUs are fast because they have thousands of cores optimized for float16/float32
matrix multiplication. A single GPU operation: multiply two floats and accumulate.

### 1-Bit Eliminates This Advantage

When weights are constrained to {-1, +1}:
- Multiplication becomes **XNOR** (1 CPU cycle)
- Accumulation becomes **popcount** (1 CPU cycle)
- No floating-point unit needed at all

CPUs can execute XNOR + popcount at billions of operations per second through
their integer ALUs. The GPU's float advantage disappears entirely.

### BitNet Paper (Wang et al., 2023)

The BitNet paper showed that transformer models can be trained with 1-bit
weights while retaining most of their quality. Key findings:
- 1-bit weights + 8-bit activations ≈ full precision quality at scale
- Energy consumption drops 71×
- Memory footprint drops 16× (1 bit vs 16 bits per weight)

We apply this to video generation, which has never been done.

---

## Previous Phases

| Phase | Repo | What It Did |
|-------|------|-------------|
| 1 | wan-profiler | Profiled where compute goes |
| 2 | mamba-video | Replaced attention with O(n) SSM |
| 3 | codec-video-gen | I-frame/P-frame temporal design |
| **4** | **bitnet-video** | **1-bit quantization** |

Each phase's speedup multiplies with the others:
- Phase 2: ~2× (attention removal)
- Phase 3: ~4-6× (temporal compression)
- Phase 4: ~8-16× (binary operations)
- **Combined: ~64-192× theoretical speedup**

---

## Hardware

- **Primary (development + benchmarking):** MacBook Air M4 — ARM64 / NEON, no GPU.
- **Supported, CI-verified:** commodity x86 with AVX2 (any modern Intel/AMD CPU).
- **Origin, proof-of-concept (retired):** Intel Pentium Gold 7505 — x86-64 / AVX2, 2C/4T, 16 GB.

CPU-native, no GPU, across **both** architectures. We develop on the M4, but **all code must stay
within the commodity-hardware design budget** — assume **2–4 cores, 16 GB RAM (~12 GB usable),
no GPU**, and it must run on x86 (AVX2) **and** ARM (NEON). The XNOR+popcount kernel must stay
architecture-portable; Phase 5's portable SIMD library implements it for both x86 (AVX2) and ARM
(NEON). The Pentium Gold proved the weakest-hardware case. Python 3.9, venv.

### Quantization-Specific Constraints:

- **All quantization must run on CPU.** No CUDA quantization kernels.
- **1-bit weights reduce model size ~16×.** A 1.3B param model at 1-bit
  ≈ 160 MB vs 2.6 GB at float16. This easily fits in 16 GB RAM.
- **Activations at 8-bit** still require float compute for the quantize/
  dequantize steps. Keep this overhead minimal.
- **The XNOR+popcount kernel is simulated** in this phase using PyTorch
  integer ops. Phase 5 (simd-kernels) provides the real native kernel (AVX2 + NEON).

---

## Code Conventions

- **Python 3.9** — `List[str]` not `list[str]`
- Type hints on all signatures, Google-style docstrings
- `logging` module, never `print()`
- `torch.no_grad()` for inference
- Tests with `pytest`, small dimensions

---

## File Structure

```
bitnet-video/
├── CLAUDE.md
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── lessons.md
├── tasks/todo.md
├── .claude/{settings.json, commands/, rules/}
├── configs/default.json
├── src/bitnet_video/
│   ├── __init__.py
│   ├── config.py          ← Quantization configuration
│   ├── quantize.py        ← Core quantization functions
│   ├── bitlinear.py       ← 1-bit Linear layer (drop-in replacement)
│   ├── bitconv.py         ← 1-bit Conv2d layer
│   ├── converter.py       ← Convert float model → bitnet model
│   ├── benchmark.py       ← Speed comparison float vs 1-bit
│   ├── quality.py         ← Quality metrics after quantization
│   ├── report.py          ← Report generation
│   └── cli.py             ← CLI entry point
├── scripts/
│   ├── run_quantize.py
│   ├── run_benchmark.py
│   └── analyze_weights.py
├── tests/
│   ├── __init__.py
│   ├── test_quantize.py
│   ├── test_bitlinear.py
│   ├── test_converter.py
│   └── test_benchmark.py
├── results/.gitkeep
└── docs/quantization_plan.md
```

---

## Key Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Quantize a model
python scripts/run_quantize.py --model wan-1.3b --output results/

# Benchmark float vs 1-bit
python scripts/run_benchmark.py --original wan-1.3b --quantized results/model_1bit/

# Analyze weight distributions
python scripts/analyze_weights.py --model wan-1.3b --output results/

# Run tests
pytest tests/ -v
```

---

## Research Questions

1. **How much quality is lost** when quantizing to 1-bit weights?
2. **What's the memory savings** (should be ~16× for weights)?
3. **How much faster** are binary matmuls on CPU (simulated)?
4. **Which layers tolerate 1-bit** and which need higher precision?
5. **Does the delta predictor quantize better** than the full model?

---

## Task & Lessons

Check `tasks/todo.md` before starting work.
Check `lessons.md` before writing new code.

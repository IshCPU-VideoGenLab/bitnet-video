# Phase 4 — bitnet-video Task Roadmap

> Claude Code: check this at the start of every session.

---

## Milestone 1: Core Quantization Functions
- [ ] Implement `weight_quant_1bit()` — sign function with scaling factor α
- [ ] Implement `activation_quant_8bit()` — symmetric 8-bit quantization
- [ ] Implement `activation_quant_absmax()` — per-tensor absmax scaling
- [ ] Handle edge cases: zero weights, all-same-sign tensors
- [ ] Unit test: quantize → dequantize roundtrip
- [ ] Unit test: scaling factor correctness
- [ ] Unit test: numerical stability with float16 inputs

## Milestone 2: BitLinear Layer
- [ ] Implement `BitLinear` as drop-in replacement for `nn.Linear`
- [ ] Forward: quantize weights → binary matmul (simulated) → scale output
- [ ] Support bias (optional)
- [ ] Straight-Through Estimator (STE) for gradient through sign()
- [ ] Unit test: output shape matches nn.Linear
- [ ] Unit test: gradient flow through STE
- [ ] Unit test: float16 forward pass stability
- [ ] Benchmark: BitLinear vs nn.Linear speed on CPU

## Milestone 3: BitConv2d Layer
- [ ] Implement `BitConv2d` as drop-in for `nn.Conv2d`
- [ ] Same quantization scheme as BitLinear but for conv weights
- [ ] Support common conv configs (stride, padding, groups)
- [ ] Unit test: shape correctness
- [ ] Unit test: gradient flow

## Milestone 4: Model Converter
- [ ] Implement `quantize_model()` — swap float layers for 1-bit
- [ ] Skip patterns: normalization, embeddings, final projection
- [ ] Report which layers were quantized vs skipped
- [ ] Compute memory savings (float vs 1-bit)
- [ ] Test: convert a small dummy model
- [ ] Test: converted model can forward pass
- [ ] Test: verify skipped layers remain in float

## Milestone 5: Speed Benchmarking
- [ ] Benchmark float nn.Linear vs BitLinear (various sizes)
- [ ] Benchmark float nn.Conv2d vs BitConv2d
- [ ] Measure memory: float model vs quantized model
- [ ] Produce speedup charts
- [ ] Note: Phase 4 uses simulated binary ops (PyTorch integer math).
       True XNOR+popcount speedup comes in Phase 5 (AVX2).

## Milestone 6: Quality Evaluation
- [ ] Quantize Wan 1.3B (or Phase 2 modified model)
- [ ] Compare outputs: float vs 1-bit (MSE, cosine, PSNR)
- [ ] Identify which layers cause most quality loss when quantized
- [ ] Test mixed-precision: keep sensitive layers in higher bits
- [ ] Quantize the delta predictor from Phase 3 separately
- [ ] Produce quality vs compression tradeoff chart

## Milestone 7: Documentation & Polish
- [ ] Write docs/quantization_plan.md
- [ ] Update README with results
- [ ] Clean code, all tests pass
- [ ] Tag v0.1.0
- [ ] Write Phase 5 (avx2-kernels) handoff summary

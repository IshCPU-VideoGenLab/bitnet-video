# Phase 4 Results — 1-bit Weight Quantization on the Real Wan DiT

Post-training 1-bit weight quantization applied to the real Wan 1.3B DiT
(CPU, bfloat16), measured by output drift over a short denoise loop.
Status: **quantization works; post-training drift is moderate (not catastrophic);
the headline win is 16× memory. Quality recovery needs QAT (or higher-precision
sensitive layers).**

## Scheme

Each linear weight is binarized with a per-output-channel scale:

```
W_bin = sign(W),   alpha = mean(|W|) per output channel,   W_q = W_bin * alpha
```

Weight-only here (a clean measurement); the full BitNet scheme also quantizes
activations to 8-bit. Norms and embeddings stay in float (standard practice).
This is **post-training** quantization (no quantization-aware training).

## Output drift (4-step denoise, vs. original)

| config | s1 | s2 | s3 | s4 (final) |
|--------|----|----|----|-----------:|
| FFN-only 1-bit (60 layers) | 0.997 | 0.986 | 0.969 | **0.947** |
| All-linear 1-bit (306 layers) | 0.990 | 0.958 | 0.907 | **0.847** |

**Finding:** post-training 1-bit degrades **moderately and gracefully** — it
compounds over steps but does not collapse. Targeting just the FFN (the 38.5%
of compute from Phase 1 Table 1) holds at 0.947; quantizing *every* linear drops
to 0.847. For comparison: self-attention Mamba surgery sits at 0.995, while
cross-attention replacement collapses to 0.37 — so 1-bit quantization is a
*recoverable* perturbation, between those extremes.

## The real win: memory

| precision | 1.42 B params |
|-----------|--------------:|
| bfloat16 (current) | ~2.84 GB |
| **1-bit + per-channel scales** | **~178 MB** (≈16×) |

This is the headline BitNet payoff: the model shrinks ~16×, fitting trivially on
cheap hardware, and matrix multiply becomes XNOR + popcount (the Phase 5 binary
GEMM kernel). Speed and the activation path are exercised separately
(simd-kernels).

## Honest caveats / next

- **Post-training, no QAT.** The 0.85–0.95 drift is what quantization-aware
  training (or keeping the most sensitive layers in higher precision) must
  recover — BitNet's published quality comes from training, not PTQ.
- **Latent cosine ≠ perceptual quality** — needs VAE decode / FID.
- Next: per-layer sensitivity (which layers tolerate 1-bit?), activation
  quantization, and wiring the binary GEMM kernel for the actual speed/memory win.

# Quantization Plan

## Objective

Apply BitNet-style 1-bit quantization to the video generation model,
reducing memory 16× and replacing float matmul with binary operations
(XNOR + popcount) that CPUs execute natively.

## The BitNet Quantization Scheme

### Weight Binarization

Each weight tensor W is quantized to:

```
W_q = sign(W) × α,  where α = mean(|W|)
```

- `sign(w)` maps each weight to {-1, +1}
- `α` preserves the overall magnitude
- Storage: 1 bit per weight + 1 scale factor per tensor

### Activation Quantization

Activations are quantized to 8-bit using symmetric absmax:

```
x_q = round(x / scale) × scale
scale = max(|x|) / 127
```

This preserves the dynamic range while reducing precision.

### Binary Matrix Multiplication

Standard matmul: `y[i] = Σ x[j] × W[i,j]`  (float multiply-accumulate)

With 1-bit weights: since W[i,j] ∈ {-1, +1}, multiplication by W
is equivalent to conditional negation. When both operands are packed
as bits:

```
y = popcount(xnor(x_bits, W_bits)) × 2 - n  (then scale by α × β)
```

This replaces all float arithmetic with integer XNOR + popcount.

## What Gets Quantized

### Quantize (1-bit weights):
- All `nn.Linear` layers in the transformer/SSM backbone
- All `nn.Conv2d` layers in the delta predictor
- Projection layers (Q, K, V projections in attention; in/out projections in SSM)

### Keep in Float:
- **Layer normalization** — needs precise mean/variance computation
- **Embeddings** — lookup tables, no matmul involved
- **Position encodings** — precision matters for spatial coherence
- **Final output layer** — quality-sensitive; last layer before decoding

## Implementation Approach

### Phase 4 (This Repo): Simulated Binary Ops

We use PyTorch's standard float matmul with binarized weights. The
quantization happens at the `forward()` call:

1. Binarize weights: `W_bin = sign(W)`
2. Compute scale: `α = mean(|W|)`
3. Standard matmul: `y = x @ W_bin.T`
4. Scale output: `y = y × α`

This is mathematically equivalent to true binary matmul but doesn't
get the speed benefit of XNOR+popcount. The purpose is to measure
**quality impact** independently from **speed optimization**.

### Phase 5 (avx2-kernels): True Binary Ops

Phase 5 replaces step 3 with actual XNOR + popcount using AVX2
intrinsics. That's where the real speedup materializes.

## Straight-Through Estimator (STE)

The sign function has zero gradient almost everywhere. For training
(or fine-tuning) with 1-bit weights, we use the Straight-Through
Estimator:

```
Forward: y = sign(x)
Backward: dy/dx = 1  (pretend sign was identity)
```

This allows gradients to flow through the quantization step, enabling
the model to learn weights that quantize well.

## Expected Results

### Memory Savings
- Float16 model: ~2.6 GB (1.3B params × 2 bytes)
- 1-bit model: ~160 MB (1.3B params × 1 bit / 8 + overhead)
- **Compression: ~16×**

### Quality Impact
Based on BitNet literature:
- 1-bit weights + 8-bit activations: ~1-3% quality degradation
- Some layers are more sensitive (first/last layers, attention)
- Mixed precision (keeping sensitive layers float) helps significantly

### Speed (Simulated)
- Phase 4 simulated binary: ~1-2× speedup (from reduced memory bandwidth)
- Phase 5 true XNOR+popcount: ~8-16× theoretical speedup

## Integration with Previous Phases

- **Phase 2 model** (Mamba SSM): Mamba's linear projections quantize
  well — they're standard matmuls. The SSM scan itself operates on
  small state vectors and stays in float.

- **Phase 3 delta predictor**: The Conv2d layers in the delta predictor
  are excellent quantization targets — small kernels, many channels.

- **Combined**: A 1-bit Mamba model generating I-frames + a 1-bit delta
  predictor generating P-frames = the full CPU-native pipeline.

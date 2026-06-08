# Python Rules — bitnet-video
- Python 3.9, List[str] not list[str], no match statements
- Type hints on ALL signatures, Google docstrings
- logging module, never print()
- torch.no_grad() for inference
- Quantization: always clamp before sign(), use float32 for scale computation
- Keep normalization and embedding layers in float — never quantize these
- Test with small dims (d=64, batch=2) first
- Files under 300 lines, absolute imports only

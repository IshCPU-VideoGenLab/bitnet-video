"""Command-line interface for bitnet-video."""

import argparse
import logging
import sys
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bitnet-video", description="1-bit quantization for video generation")
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("quantize", help="Quantize a model to 1-bit")
    q.add_argument("--model", type=str, default="wan-1.3b")
    q.add_argument("--model-path", type=str, default=None)
    q.add_argument("--output", type=str, default="results")
    q.add_argument("--activation-bits", type=int, default=8)
    q.add_argument("--debug", action="store_true")

    b = sub.add_parser("benchmark", help="Benchmark float vs 1-bit layers")
    b.add_argument("--output", type=str, default="results")
    b.add_argument("--debug", action="store_true")

    a = sub.add_parser("analyze", help="Analyze weight distributions")
    a.add_argument("--model", type=str, default="wan-1.3b")
    a.add_argument("--model-path", type=str, default=None)
    a.add_argument("--output", type=str, default="results")
    a.add_argument("--debug", action="store_true")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        print("Usage: bitnet-video {quantize|benchmark|analyze} [options]")
        return 1

    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if args.command == "quantize":
        import torch
        from transformers import AutoModel
        from bitnet_video.config import QuantConfig
        from bitnet_video.converter import quantize_model
        from bitnet_video.report import save_json, format_conversion_report

        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        path = args.model_path or args.model
        model = AutoModel.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
        model.eval()

        config = QuantConfig(activation_bits=args.activation_bits, output_dir=args.output)
        model, report = quantize_model(model, config)
        print(format_conversion_report(report))

        import os
        state_path = os.path.join(args.output, "model_1bit", "state_dict.pt")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save(model.state_dict(), state_path)
        return 0

    elif args.command == "benchmark":
        from bitnet_video.benchmark import benchmark_linear_layers
        from bitnet_video.report import save_json, format_benchmark_table

        results = benchmark_linear_layers()
        print(format_benchmark_table(results))
        data = [{"label": r.label, "float_ms": r.float_time_ms, "bit_ms": r.bit_time_ms, "speedup": r.speedup} for r in results]
        save_json(data, args.output, "benchmark_results.json")
        return 0

    elif args.command == "analyze":
        import torch
        from transformers import AutoModel
        from bitnet_video.quality import analyze_weight_distribution
        from bitnet_video.report import save_json

        path = args.model_path or args.model
        model = AutoModel.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
        results = analyze_weight_distribution(model)
        save_json(results, args.output, "weight_analysis.json")
        print(f"\nAnalyzed {len(results)} weight tensors. Results saved to {args.output}/weight_analysis.json")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

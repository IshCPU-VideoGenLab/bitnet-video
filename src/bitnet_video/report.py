"""Report generation for bitnet-video results."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def save_json(data: Any, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)
    return path


def format_conversion_report(report: Any) -> str:
    lines = [
        "", "=" * 60, "  bitnet-video Conversion Report", "=" * 60, "",
        f"  Modules inspected:     {report.total_modules}",
        f"  Linear → BitLinear:    {report.quantized_linear}",
        f"  Conv2d → BitConv2d:    {report.quantized_conv}",
        f"  Skipped (kept float):  {report.skipped}",
        "",
        f"  Original size:         {report.original_size_mb:.1f} MB",
        f"  Current size:          {report.quantized_size_mb:.1f} MB",
        f"  Theoretical 1-bit:     {report.original_size_mb / 16:.1f} MB",
        f"  Compression ratio:     {report.compression_ratio:.1f}×",
        "",
    ]
    if report.skipped_names:
        lines.append("  Skipped layers:")
        for name in report.skipped_names[:10]:
            lines.append(f"    ○ {name}")
        if len(report.skipped_names) > 10:
            lines.append(f"    ... and {len(report.skipped_names) - 10} more")
        lines.append("")
    lines.extend(["=" * 60, ""])
    return "\n".join(lines)


def format_benchmark_table(results: List[Any]) -> str:
    lines = [
        "", "=" * 70, "  BitLinear vs Linear Benchmark", "=" * 70, "",
        f"  {'Layer':<25} {'Float (ms)':>10} {'Bit (ms)':>10} {'Speedup':>10}",
        "  " + "-" * 57,
    ]
    for r in results:
        lines.append(
            f"  {r.label:<25} {r.float_time_ms:>10.2f} {r.bit_time_ms:>10.2f} {r.speedup:>10.2f}x"
        )
    lines.extend(["  " + "-" * 57, "", "=" * 70, ""])
    return "\n".join(lines)

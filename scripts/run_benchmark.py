#!/usr/bin/env python
"""Benchmark float vs 1-bit. Usage: python scripts/run_benchmark.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bitnet_video.cli import main
if __name__ == "__main__":
    sys.exit(main(["benchmark"] + sys.argv[1:]))

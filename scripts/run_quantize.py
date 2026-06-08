#!/usr/bin/env python
"""Quantize a model to 1-bit. Usage: python scripts/run_quantize.py --model wan-1.3b"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bitnet_video.cli import main
if __name__ == "__main__":
    sys.exit(main(["quantize"] + sys.argv[1:]))

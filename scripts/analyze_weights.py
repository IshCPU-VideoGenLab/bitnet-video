#!/usr/bin/env python
"""Analyze weight distributions. Usage: python scripts/analyze_weights.py --model wan-1.3b"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bitnet_video.cli import main
if __name__ == "__main__":
    sys.exit(main(["analyze"] + sys.argv[1:]))

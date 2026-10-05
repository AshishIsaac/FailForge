#!/usr/bin/env python3
"""Reproduce docs/benchmarks.md: detector latency / throughput / memory (PyTorch vs ONNX Runtime)
and ControlNet generation cost, written to benchmarks/results/bench.{json,md}.

    python benchmarks/run_all.py                    # champion of the full profile (or the reference checkpoint)
    python benchmarks/run_all.py --weights my.pt --no-synth
"""

from __future__ import annotations

import argparse
import sys

from failforge.cli import main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=None)
    p.add_argument("--no-synth", action="store_true", help="skip the (slow) ControlNet timing")
    a = p.parse_args()
    argv = ["bench"]
    if a.weights:
        argv += ["--weights", a.weights]
    if not a.no_synth:
        argv.append("--synth")
    sys.exit(main(argv))

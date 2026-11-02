#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility entry point for the canonical 26-kernel GPU benchmark.

The former implementation labeled repeated CosFormer measurements as every
Triton module. This wrapper delegates to the canonical benchmark, which
executes each explicitly implemented kernel with parity and finite-output
gates before recording timings.
"""

from __future__ import annotations

import argparse
import sys

from neuralops.benchmarks.benchmark_all_26_comprehensive import main as _canonical_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the canonical 26-kernel GPU benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kernels", default=None)
    args = parser.parse_args()

    forwarded = ["--device", args.device]
    if args.quick:
        forwarded.append("--quick")
    if args.kernels:
        forwarded.extend(["--kernels", args.kernels])
    return _canonical_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Compress Command

Implements the command-line interface for model compression, calibration, and pruning.
"""

def cmd_compress(args):
    from neuralops.core.compression_pipeline import compress_model
    print(f"Compressing model {args.model} via {args.method} (rank={args.rank})...")
    res = compress_model(args.model, method=args.method, rank=args.rank)
    print("Compression complete:", res)
    return 0

# -*- coding: utf-8 -*-
"""Multi-Seed Cross-Validation & Statistical Variance Stress Test for Seeds 100, 200, 300."""

import argparse
import sys
from neuralops.benchmarks.multi_seed_stress_test_131_72 import run_stress_test

DEFAULT_SEEDS = [100, 200, 300]


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Stress Test for Seeds 100, 200, 300")
    parser.add_argument("--output", type=str, default="results/multi_seed_stress_test_seeds_100_200_300.csv", help="Path to save CSV output")
    parser.add_argument("--output-json", type=str, default="results/multi_seed_stress_test_seeds_100_200_300.json", help="Path to save JSON output")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to execute benchmark")
    parser.add_argument("--seeds", type=str, default="100,200,300", help="Comma-separated random seeds")
    parser.add_argument("--limit-attentions", type=int, default=None, help="Limit number of attention modules")
    parser.add_argument("--limit-compressors", type=int, default=None, help="Limit number of compressor modules")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, 5 attentions, 5 compressors)")
    args = parser.parse_args()

    seeds = [100] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    lim_attn = 5 if args.quick else args.limit_attentions
    lim_comp = 5 if args.quick else args.limit_compressors

    run_stress_test(
        device_str=args.device,
        seeds=seeds,
        limit_attentions=lim_attn,
        limit_compressors=lim_comp,
        output_csv=args.output,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Unified Command‑Line Interface for SVD‑Lin Studio
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="SVD‑Lin Studio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------------------------------------------------
    # generate-data
    # -----------------------------------------------------------------
    p_gen = subparsers.add_parser("generate-data", help="Generate synthetic datasets")
    p_gen.add_argument("--type", required=True,
                       choices=["universal", "comprehensive", "advanced", "reasoning", "task_specific"],
                       help="Type of dataset to generate")
    p_gen.add_argument("--samples", type=int, default=5000, help="Number of samples (approx)")
    p_gen.add_argument("--output_dir", default="./data", help="Output directory")

    # -----------------------------------------------------------------
    # analyze-data
    # -----------------------------------------------------------------
    p_ana = subparsers.add_parser("analyze-data", help="Analyse dataset distribution")
    p_ana.add_argument("--data_dir", default="./data", help="Directory containing CSV files")
    p_ana.add_argument("--output", default="distribution_report.png", help="Output image file")

    # -----------------------------------------------------------------
    # benchmark-speed
    # -----------------------------------------------------------------
    p_bs = subparsers.add_parser("benchmark-speed", help="Benchmark inference speed")
    p_bs.add_argument("--original", help="Path to original model")
    p_bs.add_argument("--compressed", help="Path to compressed model")
    p_bs.add_argument("--rank", type=int, default=32, help="SVD rank (if models not given)")
    p_bs.add_argument("--model_name", default="bert-base-uncased", help="Base model name")
    p_bs.add_argument("--texts", nargs="+", default=["This is a test sentence."], help="Test texts")

    # -----------------------------------------------------------------
    # compress
    # -----------------------------------------------------------------
    p_comp = subparsers.add_parser("compress", help="Compress a model")
    p_comp.add_argument("--model", default="bert-base-uncased")
    p_comp.add_argument("--rank", type=int, default=32)
    p_comp.add_argument("--method", default="standard_svd")
    p_comp.add_argument("--finetune", action="store_true")
    p_comp.add_argument("--task", default="cola")
    p_comp.add_argument("--output", help="Output directory")

    # -----------------------------------------------------------------
    # ensemble
    # -----------------------------------------------------------------
    p_ens = subparsers.add_parser("ensemble", help="Run weighted ensemble")
    p_ens.add_argument("--dataset", required=True, help="Path to CSV")

    # -----------------------------------------------------------------
    # router
    # -----------------------------------------------------------------
    p_rou = subparsers.add_parser("router", help="Run smart router")
    p_rou.add_argument("--dataset", default="./data", help="Data directory")

    # -----------------------------------------------------------------
    # benchmark
    # -----------------------------------------------------------------
    p_ben = subparsers.add_parser("benchmark", help="Run full compression benchmark")
    p_ben.add_argument("--epochs", type=int, default=1, help="Quick test epochs")
    p_ben.add_argument("--batch_size", type=int, default=8)

    # -----------------------------------------------------------------
    # soa
    # -----------------------------------------------------------------
    p_soa = subparsers.add_parser("soa", help="Build SOA model")
    p_soa.add_argument("--model", default="bert-base-uncased")
    p_soa.add_argument("--rank", type=int, default=64)
    p_soa.add_argument("--num_labels", type=int, default=2)

    # -----------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------
    args = parser.parse_args()

    if args.command == "generate-data":
        from .data.synthetic import run_universal_generation, run_comprehensive_generation, run_advanced_generation
        from .data.synthetic_advanced import generate_reasoning_datasets
        from .data.synthetic_task_specific import run_all_task_generations

        out = Path(args.output_dir)
        if args.type == "universal":
            run_universal_generation(out, total_samples_per_domain=args.samples)
        elif args.type == "comprehensive":
            run_comprehensive_generation(out, total_samples=args.samples)
        elif args.type == "advanced":
            run_advanced_generation(out, total_samples=args.samples)
        elif args.type == "reasoning":
            generate_reasoning_datasets(out, samples_per_task=args.samples)
        elif args.type == "task_specific":
            run_all_task_generations(out)

    elif args.command == "analyze-data":
        from .data.analyze_datasets import plot_distribution_report
        plot_distribution_report(data_dir=args.data_dir, output_image=args.output)

    elif args.command == "benchmark-speed":
        from .evaluation.speed_benchmark import run_speed_benchmark
        run_speed_benchmark(original_model_path=args.original,
                            compressed_model_path=args.compressed,
                            rank=args.rank,
                            model_name=args.model_name,
                            test_texts=args.texts)

    elif args.command == "compress":
        from .cli_actions import cmd_compress
        cmd_compress(args)

    elif args.command == "ensemble":
        from .cli_actions import cmd_ensemble
        cmd_ensemble(args)

    elif args.command == "router":
        from .cli_actions import cmd_router
        cmd_router(args)

    elif args.command == "benchmark":
        from .cli_actions import cmd_benchmark
        cmd_benchmark(args)

    elif args.command == "soa":
        from .cli_actions import cmd_soa
        cmd_soa(args)

if __name__ == "__main__":
    main()
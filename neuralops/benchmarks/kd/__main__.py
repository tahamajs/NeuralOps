"""Command-line interface entry point for knowledge distillation benchmark.

Examples
--------
$ python -m neuralops.benchmarks.kd --task glue:sst2 --kd_steps 200
$ python -m neuralops.benchmarks.kd --task glue:sst2 --no-resume
"""

from __future__ import annotations

import argparse
import logging
import sys

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.runner import KDRunner


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments into a BenchmarkConfig instance.

    Returns
    -------
    BenchmarkConfig
        Resolved benchmark configuration dataclass.
    """
    parser = argparse.ArgumentParser(
        description="Run Knowledge Distillation (KD) benchmark across attention modules."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="glue:sst2",
        help="Dataset specification (e.g. 'glue:sst2', 'imdb').",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="textattack/bert-base-uncased-SST-2",
        help="Pretrained teacher model ID or path.",
    )
    parser.add_argument(
        "--pre_kd_samples",
        type=int,
        default=200,
        help="Validation sample count for pre-KD screening.",
    )
    parser.add_argument(
        "--post_kd_samples",
        type=int,
        default=872,
        help="Validation sample count for post-KD evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--kd_steps",
        type=int,
        default=200,
        help="Number of distillation training steps.",
    )
    parser.add_argument(
        "--kd_lr",
        type=float,
        default=1e-5,
        help="Learning rate for student attention parameters.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="Interval of steps for validation during KD.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature scaling.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight of soft teacher KL loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/kd_benchmark",
        help="Directory to save benchmarks and reports.",
    )
    parser.add_argument(
        "--module_whitelist",
        nargs="+",
        default=None,
        help="Space-separated list of attention module names to benchmark.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force fresh start and ignore previous state file.",
    )
    parser.set_defaults(resume=True)

    args = parser.parse_args()
    return BenchmarkConfig(
        task=args.task,
        teacher=args.teacher,
        pre_kd_samples=args.pre_kd_samples,
        post_kd_samples=args.post_kd_samples,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_length=args.max_length,
        kd_steps=args.kd_steps,
        kd_lr=args.kd_lr,
        eval_every=args.eval_every,
        temperature=args.temperature,
        alpha=args.alpha,
        output_dir=args.output_dir,
        resume=args.resume,
        module_whitelist=args.module_whitelist,
    )


def main() -> None:
    """CLI execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = parse_args()
    runner = KDRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

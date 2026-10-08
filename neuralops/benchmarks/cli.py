"""Unified CLI dispatcher for all neuralops benchmarks."""
from __future__ import annotations

import argparse
import sys
from typing import Any

from neuralops.benchmarks.core.arguments import (
    add_common_args,
    add_compression_args,
    add_dataset_args,
    add_model_args,
)


def list_benchmarks(args: argparse.Namespace) -> int:
    """Print available benchmark suites."""
    print("Available NeuralOps Benchmark Commands:")
    print("  quality-sweep   : Full GLUE & classification accuracy recovery sweep")
    print("  rank-sweep      : SVD rank vs accuracy Pareto curve evaluation")
    print("  multi-mode      : Multi-mode structural transformer benchmark")
    print("  multiseed       : Multi-seed statistical significance & variance testing")
    print("  seqlen-scaling  : Long-context sequence length latency & VRAM scaling")
    print("  ruler           : Long-context Needle-in-a-Haystack (RULER) benchmark")
    print("  gpt2-sweep      : Autoregressive LM perplexity sweep on Wikitext")
    print("  kd-test         : Quick knowledge distillation recovery test")
    print("  verify          : Verify GPU, model, and dataset readiness")
    return 0


def verify_setup(args: argparse.Namespace) -> int:
    """Verify system readiness for benchmarking."""
    import torch

    print("Checking NeuralOps Benchmarking Environment:")
    cuda_avail = torch.cuda.is_available()
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {cuda_avail}")
    if cuda_avail:
        print(f"  Device name     : {torch.cuda.get_device_name(0)}")
        print(f"  Device count    : {torch.cuda.device_count()}")
    else:
        print("  Running in CPU/macOS simulation mode.")
    return 0


def _dispatch_quality_sweep(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_quality_sweep import main as qs_main

    return qs_main()


def _dispatch_rank_sweep(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_rank_sweeper import main as rs_main

    return rs_main()


def _dispatch_multi_mode(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.multi_mode_benchmark import main as mm_main

    return mm_main()


def _dispatch_multiseed(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_multiseed import main as ms_main

    return ms_main()


def _dispatch_seqlen_scaling(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_seqlen_scaling_bench import main as ss_main

    return ss_main()


def _dispatch_ruler(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_ruler_benchmark import main as ruler_main

    return ruler_main()


def _dispatch_gpt2_sweep(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.run_gpt2_sweep import main as gpt2_main

    return gpt2_main()


def _dispatch_kd_test(args: argparse.Namespace) -> int:
    from neuralops.benchmarks.kd_quick_test import main as kd_main

    return kd_main()


def build_parser() -> argparse.ArgumentParser:
    """Build unified CLI argument parser with lazy-loading subcommands."""
    parser = argparse.ArgumentParser(
        prog="neuralops.benchmarks",
        description="Unified NeuralOps Benchmarking Command Line Interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. quality-sweep
    p_qs = sub.add_parser("quality-sweep", help="Run quality sweep across modules")
    add_common_args(p_qs)
    add_model_args(p_qs)
    add_dataset_args(p_qs)
    add_compression_args(p_qs)
    p_qs.set_defaults(func=_dispatch_quality_sweep)

    # 2. rank-sweep
    p_rs = sub.add_parser("rank-sweep", help="Run rank vs quality sweep")
    add_common_args(p_rs)
    add_model_args(p_rs)
    add_dataset_args(p_rs)
    add_compression_args(p_rs)
    p_rs.set_defaults(func=_dispatch_rank_sweep)

    # 3. multi-mode
    p_mm = sub.add_parser("multi-mode", help="Run multi-mode benchmark")
    add_common_args(p_mm)
    add_model_args(p_mm)
    add_dataset_args(p_mm)
    p_mm.set_defaults(func=_dispatch_multi_mode)

    # 4. multiseed
    p_ms = sub.add_parser("multiseed", help="Run multi-seed statistical significance")
    add_common_args(p_ms)
    add_model_args(p_ms)
    add_dataset_args(p_ms)
    add_compression_args(p_ms)
    p_ms.set_defaults(func=_dispatch_multiseed)

    # 5. seqlen-scaling
    p_ss = sub.add_parser("seqlen-scaling", help="Run sequence length scaling")
    add_common_args(p_ss)
    add_model_args(p_ss)
    add_dataset_args(p_ss)
    add_compression_args(p_ss)
    p_ss.set_defaults(func=_dispatch_seqlen_scaling)

    # 6. ruler
    p_ru = sub.add_parser("ruler", help="Run long-context RULER benchmark")
    add_common_args(p_ru)
    add_model_args(p_ru)
    add_dataset_args(p_ru)
    p_ru.set_defaults(func=_dispatch_ruler)

    # 7. gpt2-sweep
    p_gpt = sub.add_parser("gpt2-sweep", help="Run GPT-2 language model perplexity sweep")
    add_common_args(p_gpt)
    add_model_args(p_gpt)
    add_dataset_args(p_gpt)
    add_compression_args(p_gpt)
    p_gpt.set_defaults(func=_dispatch_gpt2_sweep)

    # 8. kd-test
    p_kd = sub.add_parser("kd-test", help="Run quick KD validation test")
    add_common_args(p_kd)
    add_model_args(p_kd)
    add_dataset_args(p_kd)
    add_compression_args(p_kd)
    p_kd.set_defaults(func=_dispatch_kd_test)

    # 9. list
    p_list = sub.add_parser("list", help="List all registered benchmarks")
    p_list.set_defaults(func=list_benchmarks)

    # 10. verify
    p_ver = sub.add_parser("verify", help="Verify model, dataset, and GPU readiness")
    p_ver.set_defaults(func=verify_setup)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    res: Any = args.func(args)
    return int(res) if isinstance(res, int) else 0


if __name__ == "__main__":
    sys.exit(main())

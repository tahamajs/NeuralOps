#!/usr/bin/env python3
"""Run NIAH benchmark for multiple modules sequentially."""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("NIAH-Seq")

HERE = Path(__file__).resolve().parent.parent


def run_niah(module_name, args, timeout_s=1800):
    cmd = [
        sys.executable, "-u", "neuralops/benchmarks/run_ruler_benchmark.py",
        "--module", module_name,
        "--model_path", args.model_path,
        "--rank", str(args.rank),
        "--context_lengths", *[str(c) for c in args.context_lengths],
        "--num_samples", str(args.num_samples),
        "--depths", *[str(d) for d in args.depths],
        "--device", args.device,
        "--max_position", str(args.max_position),
        "--output_dir", args.output_dir,
    ]
    if args.smoke_test:
        cmd.append("--smoke_test")
    if module_name == "full":
        cmd.append("--no_inject")

    LOG.info("=" * 80)
    LOG.info("Running NIAH for module: %s", module_name)
    LOG.info("Command: %s", " ".join(cmd))
    LOG.info("=" * 80)

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=str(HERE), timeout=timeout_s, capture_output=False)
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            LOG.info("[%s] DONE in %.1fs", module_name, elapsed)
            return "PASS"
        LOG.error("[%s] FAILED (code %d) in %.1fs", module_name, result.returncode, elapsed)
        return "FAIL"
    except subprocess.TimeoutExpired:
        LOG.error("[%s] TIMEOUT after %.1fs", module_name, time.monotonic() - t0)
        return "TIMEOUT"
    except Exception as e:
        LOG.error("[%s] CRASH: %s", module_name, e)
        return "CRASH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", nargs="+", required=True)
    parser.add_argument("--model_path", default="gpt2")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_position", type=int, default=1024)
    parser.add_argument("--output_dir", default="results/06_adhoc_experiments/ruler_benchmark")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=1800)

    args = parser.parse_args()

    LOG.info("=" * 80)
    LOG.info("NIAH SEQUENTIAL — %d modules: %s", len(args.modules), args.modules)
    LOG.info("Contexts: %s | Depths: %s | Samples: %d", args.context_lengths, args.depths, args.num_samples)
    LOG.info("=" * 80)

    results = {}
    t_start = time.monotonic()
    for i, module_name in enumerate(args.modules, 1):
        LOG.info("\n>>> [%d/%d] %s", i, len(args.modules), module_name)
        status = run_niah(module_name, args, timeout_s=args.timeout)
        results[module_name] = status
        time.sleep(5)

    total_time = time.monotonic() - t_start
    LOG.info("\n" + "=" * 80)
    LOG.info("SUMMARY (%.1f min)", total_time / 60)
    LOG.info("=" * 80)
    for m, s in results.items():
        LOG.info("%-40s | %s", m, s)
    LOG.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
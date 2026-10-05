#!/usr/bin/env python3
"""Run kd_quick_test.py for multiple modules sequentially."""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("KD-All")

HERE = Path(__file__).resolve().parent


def read_modules_from_file(path="all_modules.txt"):
    """Read module list from all_modules.txt."""
    f = Path(path)
    if not f.exists():
        LOG.error("File %s not found. Run list_modules.py first.", path)
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def run_module(module_name, args, timeout_s=3600):
    """Run kd_quick_test for a single module."""
    cmd = [
        sys.executable, "-u", "-m", "neuralops.benchmarks.kd_quick_test",
        "--module", module_name,
        "--model_name", args.model_name,
        "--rank", str(args.rank),
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--recovery_steps", str(args.recovery_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--eval_every", str(args.eval_every),
        "--methods", *args.methods,
        "--device", args.device,
        "--train_scope", args.train_scope,
        "--max_train_texts", str(args.max_train_texts),
    ]
    if args.baseline_ppl is not None:
        cmd.extend(["--baseline_ppl", str(args.baseline_ppl)])
    if args.fp16:
        cmd.append("--fp16")
    if args.no_cosine_lr:
        cmd.append("--no_cosine_lr")
    if args.use_cache:
        cmd.append("--use_cache")
    if args.cache_attn:
        cmd.append("--cache_attn")
    if args.num_chunks > 0:
        cmd.extend(["--num_chunks", str(args.num_chunks)])

    LOG.info("Running: %s", " ".join(cmd))

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(HERE.parent),
            timeout=timeout_s,
            capture_output=False,
        )
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            LOG.info("[%s] DONE in %.1fs", module_name, elapsed)
            return "PASS"
        else:
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

    # Module selection
    parser.add_argument("--modules", nargs="+", default=None,
                        help="List of modules. If None, reads from all_modules.txt")
    parser.add_argument("--modules_file", default="all_modules.txt")
    parser.add_argument("--skip", nargs="+", default=[])

    # Model & data
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--max_train_texts", type=int, default=5000)
    parser.add_argument("--num_chunks", type=int, default=412)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--recovery_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--no_cosine_lr", action="store_true")

    # KD
    parser.add_argument("--methods", nargs="+", default=["C"],
                        choices=["A", "B", "C", "D"])

    # Cache
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--cache_attn", action="store_true")

    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--train_scope", default="module",
                        choices=["all", "mamba", "attention", "module"])
    parser.add_argument("--baseline_ppl", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=1800)

    args = parser.parse_args()

    # Build module list
    if args.modules:
        modules = args.modules
    else:
        modules = read_modules_from_file(args.modules_file)
        if not modules:
            LOG.error("No modules to run. Aborting.")
            return 1

    # Apply skip
    if args.skip:
        skip_set = set(args.skip)
        modules = [m for m in modules if m not in skip_set]

    # Deduplicate and skip already run modules
    import json
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY
        has_reg = True
    except ImportError:
        has_reg = False

    completed = set()
    reg_file = HERE.parent / "results" / "kd_runs" / "registry.json"
    if reg_file.exists():
        try:
            data = json.loads(reg_file.read_text(encoding="utf-8"))
            for run_entry in data.get("runs", []):
                if "module" in run_entry:
                    completed.add(run_entry["module"])
        except Exception as e:
            LOG.warning(f"Could not read registry.json: {e}")

    final_modules = []
    seen_refs = set()
    for m in modules:
        if m in completed:
            LOG.info(f"Skipping {m} (already evaluated in registry.json)")
            continue
        
        if has_reg:
            try:
                cls = ATTENTION_REGISTRY.resolve(m)
                if id(cls) in seen_refs:
                    LOG.info(f"Skipping {m} (alias of already tested class)")
                    continue
                seen_refs.add(id(cls))
            except Exception:
                pass
                
        final_modules.append(m)

    modules = final_modules
    
    if not modules:
        LOG.error("All modules skipped. Aborting.")
        return 0

    LOG.info("=" * 80)
    LOG.info("KD QUICK TEST — %d modules", len(modules))
    LOG.info("Model: %s", args.model_name)
    LOG.info("Methods: %s", args.methods)
    LOG.info("Recovery steps: %d", args.recovery_steps)
    LOG.info("=" * 80)

    results = {}
    t_start = time.monotonic()

    for i, module_name in enumerate(modules, 1):
        LOG.info("\n" + "=" * 80)
        LOG.info(">>> [%d/%d] %s", i, len(modules), module_name)
        LOG.info("=" * 80)
        status = run_module(module_name, args, timeout_s=args.timeout)
        results[module_name] = status

    total_time = time.monotonic() - t_start

    # Summary
    LOG.info("\n" + "=" * 80)
    LOG.info("SUMMARY (%.1f min total)", total_time / 60)
    LOG.info("=" * 80)
    for m, s in results.items():
        LOG.info("%-50s | %s", m, s)
    LOG.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
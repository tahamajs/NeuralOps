#!/usr/bin/env python3
"""
Run NIAH benchmark on ALL Tier-1 modules sequentially.
Reads module list from results/audit/trusted_modules.json.
Handles crashes gracefully and continues to next module.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("NIAH-All")

HERE = Path(__file__).resolve().parent.parent


def load_tier1_modules():
    """Load Tier-1 modules from trusted_modules.json."""
    path = HERE / "results" / "audit" / "trusted_modules.json"
    if not path.exists():
        LOG.error("trusted_modules.json not found at %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    modules = data.get("tier1_true_causal", [])
    # Remove duplicates and sort
    return sorted(set(modules))


def dedupe_modules(modules):
    """Remove obvious aliases (keep canonical names)."""
    # Prefer names without _attention, _attn suffixes when both exist
    canonical = {}
    for m in modules:
        base = m
        for suf in ["_attention", "_attn", "_module", "_class"]:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        canonical.setdefault(base, []).append(m)
    
    result = []
    for base, variants in canonical.items():
        # Prefer the shortest name (usually canonical)
        variants.sort(key=lambda x: (len(x), x))
        result.append(variants[0])
    return sorted(set(result))


def run_niah(module_name, args, timeout_s=600):
    """Run NIAH for a single module."""
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
    LOG.info("Running NIAH for: %s", module_name)
    LOG.info("=" * 80)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, cwd=str(HERE), timeout=timeout_s, capture_output=True, text=True,
        )
        elapsed = time.monotonic() - t0
        # Print tail of stdout for visibility
        if result.stdout:
            for line in result.stdout.splitlines()[-20:]:
                LOG.info("  | %s", line)
        if result.returncode == 0:
            LOG.info("[%s] ✓ DONE in %.1fs", module_name, elapsed)
            return "PASS"
        LOG.error("[%s] ✗ FAILED (code %d) in %.1fs", module_name, result.returncode, elapsed)
        if result.stderr:
            for line in result.stderr.splitlines()[-5:]:
                LOG.error("  | %s", line)
        return "FAIL"
    except subprocess.TimeoutExpired:
        LOG.error("[%s] ⏱️ TIMEOUT after %.1fs", module_name, timeout_s)
        return "TIMEOUT"
    except Exception as e:
        LOG.error("[%s] 💥 CRASH: %s", module_name, e)
        return "CRASH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", nargs="+", default=None,
                        help="Explicit modules. If None, reads from trusted_modules.json")
    parser.add_argument("--smoke_test", action="store_true", default=True,
                        help="Use smoke test (4 samples, ctx=512 only)")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Full test (20 samples, ctx=512+1024)")
    parser.add_argument("--model_path", default="gpt2")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[512])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_position", type=int, default=1024)
    parser.add_argument("--output_dir", default="results/06_adhoc_experiments/ruler_benchmark")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Modules to skip")

    args = parser.parse_args()

    # If --full, override smoke defaults
    if args.full:
        args.smoke_test = False
        args.context_lengths = [512, 1024]
        args.num_samples = 20

    # Build module list
    if args.modules:
        modules = list(args.modules)
    else:
        modules = load_tier1_modules()
        modules = dedupe_modules(modules)
        LOG.info("Loaded %d unique Tier-1 modules", len(modules))

    # Apply skip
    if args.skip:
        skip_set = set(args.skip)
        modules = [m for m in modules if m not in skip_set]

    # Always run baseline first
    if "full" not in modules:
        modules = ["full"] + modules

    LOG.info("=" * 80)
    LOG.info("NIAH ALL MODULES — %d modules", len(modules))
    LOG.info("Mode: %s", "FULL" if args.full else "SMOKE")
    LOG.info("Contexts: %s | Samples: %d | Depths: %s",
             args.context_lengths, args.num_samples, args.depths)
    LOG.info("=" * 80)

    # Save list for reference
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_modules_to_run.txt").write_text("\n".join(modules))

    results = {}
    t_start = time.monotonic()

    for i, module_name in enumerate(modules, 1):
        LOG.info("\n" + "=" * 80)
        LOG.info(">>> [%d/%d] %s", i, len(modules), module_name)
        LOG.info("=" * 80)

        status = run_niah(module_name, args, timeout_s=args.timeout)
        results[module_name] = status
        time.sleep(3)  # Cool down

    # Summary
    total_time = time.monotonic() - t_start
    LOG.info("\n" + "=" * 80)
    LOG.info("FINAL SUMMARY (%.1f min)", total_time / 60)
    LOG.info("=" * 80)
    passed = sum(1 for s in results.values() if s == "PASS")
    failed = sum(1 for s in results.values() if s == "FAIL")
    timeout = sum(1 for s in results.values() if s == "TIMEOUT")
    crash = sum(1 for s in results.values() if s == "CRASH")

    LOG.info("PASS:    %d", passed)
    LOG.info("FAIL:    %d", failed)
    LOG.info("TIMEOUT: %d", timeout)
    LOG.info("CRASH:   %d", crash)
    LOG.info("")
    LOG.info("%-45s | %s", "Module", "Status")
    LOG.info("-" * 60)
    for m, s in results.items():
        LOG.info("%-45s | %s", m, s)

    # Save final report
    (out / "_niah_all_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    LOG.info("\nResults saved to %s", out / "_niah_all_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
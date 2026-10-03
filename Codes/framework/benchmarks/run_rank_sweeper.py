#!/usr/bin/env python3
"""
Automated rank sweep runner for low-rank compressor modules.

Features:
- Sequential OR parallel rank execution (--parallel_ranks N)
- Idempotent: skips ranks whose summary.csv already has a final status
- --fresh to force re-run
- Auto-detects parent_checkpoint and cached_batches_dir
- Aggregated CSV across all modules (all_rank_curves.csv)
- Graceful Ctrl+C handling
- Cross-platform (Windows + Linux)
- Per-rank log file in each rank_dir/run.log
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================
DEFAULT_RANKS = [16, 32, 64, 128, 256, 384]

# Statuses that indicate a final evaluation is done -> don't re-run
SKIP_IF_STATUS_IN = {
    "PASS", "PASS_LOW", "PASS_LOW_PPL", "PASS_NO_COMPRESSION",
    "FAIL", "FAIL_BELOW_CHANCE", "FAIL_PPL_NAN", "FAIL_PPL_DIVERGED",
    "EVAL_FAILED", "INVALID",
    "EXTREME_EXPANSION", "MODERATE_EXPANSION",
    "SKIP_BLOCKED", "SKIP_DEPRECATED",
    "SKIP_BERT_ARCHITECTURAL",
}

_print_lock = threading.Lock()
_interrupted = threading.Event()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ============================================================================
# Arg parsing
# ============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rank sweep for compressor modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modules", required=True,
                   help="Comma-separated list of modules to sweep")
    p.add_argument("--ranks", default=",".join(str(r) for r in DEFAULT_RANKS),
                   help="Comma-separated ranks")
    p.add_argument("--model", default="models/distilbert_sst2",
                   help="Model path")
    p.add_argument("--model_family", default="bert",
                   help="Model family (bert/roberta/gpt2/qwen)")
    p.add_argument("--dataset", default="sst2",
                   help="Dataset name")
    p.add_argument("--recovery_steps", type=int, default=50,
                   help="Recovery fine-tuning steps")
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="Learning rate for recovery")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size (None = auto)")
    p.add_argument("--parallel_ranks", type=int, default=1,
                   help="Number of ranks to run in parallel")
    p.add_argument("--max_vram_fraction", type=float, default=None,
                   help="Max VRAM fraction per process (default: auto)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Timeout per rank in seconds")
    p.add_argument("--output_base", default="results/04_rank_sweeps/mixed/rank_sweep",
                   help="Base output directory")
    p.add_argument("--parent_checkpoint", default=None,
                   help="Path to baseline checkpoint (auto-detect if None)")
    p.add_argument("--cached_batches_dir", default=None,
                   help="Batch cache dir (auto-detect if None)")
    p.add_argument("--fresh", action="store_true",
                   help="Force re-run all ranks")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running")
    return p.parse_args(argv)


# ============================================================================
# Helpers
# ============================================================================
def read_summary_row(summary_csv: Path) -> Optional[Dict[str, str]]:
    if not summary_csv.exists():
        return None
    try:
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception as e:
        log(f"[WARN] Could not read {summary_csv}: {e}")
    return None


def auto_detect_parent_checkpoint(model: str, dataset: str) -> Optional[str]:
    safe_model = Path(model).name.replace("-", "_").lower()
    candidates = [
        Path("results") / f"{safe_model}_{dataset}_baseline" / "finetuned_model.pt",
        Path("results") / f"{safe_model}_{dataset}_lora_baseline" / "finetuned_model.pt",
        Path("results") / f"{safe_model}_{dataset}_full_sweep" / "finetuned_model.pt",
        Path(model) / "finetuned_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def auto_detect_cached_batches(model: str, dataset: str) -> Optional[str]:
    safe_model = Path(model).name.replace("-", "_").lower()
    candidates = [
        Path("results") / f"{safe_model}_{dataset}_baseline" / f"batch_cache_{dataset}",
        Path("results") / f"{safe_model}_{dataset}_lora_baseline" / f"batch_cache_{dataset}",
        Path("results") / f"{safe_model}_{dataset}_full_sweep" / f"batch_cache_{dataset}",
    ]
    for c in candidates:
        if c.exists() and (c / "meta.json").exists():
            return str(c)
    return None


def build_cmd(module: str, rank: int, args: argparse.Namespace,
              rank_dir: Path, vram_fraction: Optional[float]) -> List[str]:
    rank_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        "Codes/framework/run_quality_sweep.py",
        "--modules", module,
        "--model", args.model,
        "--model_family", args.model_family,
        "--dataset", args.dataset,
        "--rank", str(rank),
        "--recovery_steps", str(args.recovery_steps),
        "--learning_rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--output_dir", str(rank_dir) + os.sep,
        "--parallel", "1",          # internal parallelism is 1; we parallelize at rank level
        "--timeout", str(args.timeout),
    ]
    if args.fresh:
        cmd.append("--fresh")
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.parent_checkpoint and Path(args.parent_checkpoint).exists():
        cmd.extend(["--parent_checkpoint", args.parent_checkpoint])
    if args.cached_batches_dir and Path(args.cached_batches_dir).exists():
        cmd.extend(["--cached_batches_dir", args.cached_batches_dir])
    if vram_fraction is not None:
        cmd.extend(["--max_vram_fraction", str(vram_fraction)])
    return cmd


def run_single_rank(module: str, rank: int, args: argparse.Namespace,
                    vram_fraction: Optional[float]) -> Dict[str, Any]:
    """Run one (module, rank) evaluation. Returns a result row dict."""
    rank_dir = Path(args.output_base) / module / f"rank_{rank}"
    summary_csv = rank_dir / "summary.csv"

    # Idempotent skip
    if not args.fresh and summary_csv.exists():
        row = read_summary_row(summary_csv)
        if row and row.get("status") in SKIP_IF_STATUS_IN:
            log(f"[{module} | r{rank}] SKIP — existing "
                f"status={row.get('status')} acc={row.get('accuracy_pct', '?')}%")
            out = dict(row)
            out["rank"] = rank
            return out

    cmd = build_cmd(module, rank, args, rank_dir, vram_fraction)
    log(f"[{module} | r{rank}] LAUNCH")

    if args.dry_run:
        log(f"    (dry-run) {' '.join(cmd)}")
        return {"module_name": module, "rank": rank, "status": "DRY_RUN",
                "accuracy_pct": 0.0, "elapsed_s": 0.0}

    log_file = rank_dir / "run.log"
    t0 = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            res = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        elapsed = time.time() - t0
        rc = res.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"[{module} | r{rank}] TIMEOUT after {elapsed:.1f}s")
        return _error_row(module, rank, f"TIMEOUT_{args.timeout}s", elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        log(f"[{module} | r{rank}] EXCEPTION: {e}")
        return _error_row(module, rank, f"EXCEPTION:{e}", elapsed)

    if rc != 0:
        log(f"[{module} | r{rank}] non-zero exit code {rc} ({elapsed:.1f}s)")

    row = read_summary_row(summary_csv)
    if row:
        out = dict(row)
        out["rank"] = rank
        out["elapsed_s"] = round(elapsed, 1)
        log(f"[{module} | r{rank}] DONE in {elapsed:.1f}s — "
            f"status={out.get('status', '?')} acc={out.get('accuracy_pct', '?')}%")
        return out

    log(f"[{module} | r{rank}] NO summary.csv produced (rc={rc}, {elapsed:.1f}s)")
    return _error_row(module, rank, f"ERROR_CODE_{rc}", elapsed)


def _error_row(module: str, rank: int, status: str, elapsed: float) -> Dict[str, Any]:
    return {
        "module_name": module, "rank": rank, "status": status,
        "accuracy_pct": 0.0, "f1_score": 0.0, "val_loss": 99.0,
        "peak_vram_mb": 0.0, "compression_rate": 1.0,
        "num_compressed_layers": 0, "num_fallback_layers": 0,
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================================
# Output helpers
# ============================================================================
CURVE_FIELDS = [
    "rank", "module_name", "status", "accuracy_pct", "f1_score", "val_loss",
    "peak_vram_mb", "compression_rate", "compression_ratio_pct",
    "num_compressed_layers", "num_fallback_layers", "elapsed_s",
    "total_params_before", "total_params_after",
]


def auto_git_commit(module: str, output_base: str) -> None:
    """Auto-commit completed module results, logs, and curves into git."""
    try:
        module_dir = Path(output_base) / module
        # Stage results for this module
        subprocess.run(
            ["git", "add", str(module_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        msg = f"benchmarks({module}): multi-rank sweep logs and curves [{Path(output_base).name}]"
        subprocess.run(
            ["git", "commit", "-m", msg, "--no-verify"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        log(f"[{module}] Committed results to git log.")
    except Exception as e:
        log(f"[{module}] Git auto-commit notice: {e}")


def write_rank_curve(module: str, rows: List[Dict[str, Any]], output_base: str) -> Path:
    module_dir = Path(output_base) / module
    module_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = module_dir / "rank_curve.csv"
    rows_sorted = sorted(rows, key=lambda r: int(r.get("rank", 0)))
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    log(f"[{module}] Wrote {curve_csv}")
    auto_git_commit(module, output_base)
    return curve_csv


def write_combined_curve(all_rows: List[Dict[str, Any]], output_base: str) -> Path:
    out_path = Path(output_base) / "all_rank_curves.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    log(f"[ALL] Wrote combined {out_path}")
    try:
        subprocess.run(["git", "add", str(out_path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        subprocess.run(["git", "commit", "-m", f"benchmarks: update all_rank_curves.csv [{Path(output_base).name}]", "--no-verify"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        pass
    return out_path


def print_rank_table(module: str, rows: List[Dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r.get("rank", 0)))
    print(f"\n### Rank Sweep Results: `{module}`")
    print("| Rank | Status | Acc (%) | F1 | Val Loss | VRAM (MB) | Comp Rate | Layers (C/F) | Time (s) |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    def _f(d, k, default=0.0):
        try:
            v = d.get(k, default)
            return float(v) if v not in (None, "") else default
        except (ValueError, TypeError):
            return default

    for r in rows_sorted:
        print(
            f"| {r.get('rank')} | {r.get('status', '?')} | "
            f"{_f(r, 'accuracy_pct'):.2f} | {_f(r, 'f1_score'):.4f} | "
            f"{_f(r, 'val_loss'):.4f} | {_f(r, 'peak_vram_mb'):.1f} | "
            f"{_f(r, 'compression_rate', 1.0):.2f}x | "
            f"{r.get('num_compressed_layers', 0)}/{r.get('num_fallback_layers', 0)} | "
            f"{_f(r, 'elapsed_s'):.1f} |"
        )


def print_summary(all_rows: List[Dict[str, Any]]) -> None:
    if not all_rows:
        return
    from collections import defaultdict
    by_mod: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_mod[r.get("module_name", "?")].append(r)
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for mod, rows in sorted(by_mod.items()):
        passed = [r for r in rows if r.get("status") in ("PASS", "PASS_LOW", "PASS_LOW_PPL", "PASS_NO_COMPRESSION")]
        total = len(rows)
        if passed:
            best = max(passed, key=lambda r: float(r.get("accuracy_pct") or 0.0))
            print(f"  {mod:<32} {len(passed)}/{total} PASS  | "
                  f"best: rank={best.get('rank')} acc={best.get('accuracy_pct')}% "
                  f"comp={best.get('compression_rate', 1.0)}x")
        else:
            print(f"  {mod:<32} {len(passed)}/{total} PASS")


# ============================================================================
# Signal handling
# ============================================================================
def _install_sigint_handler() -> None:
    def handler(signum, frame):
        if _interrupted.is_set():
            log("\n[FATAL] Second Ctrl+C — exiting immediately.")
            os._exit(130)
        _interrupted.set()
        log("\n[WARN] Ctrl+C — waiting for currently-running ranks to finish...")
    try:
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    # Ensure we're running from project root
    if not Path("Codes/framework/run_quality_sweep.py").exists():
        print("[FATAL] Must run from the project root directory. "
              "Could not find Codes/framework/run_quality_sweep.py")
        return 2

    args = parse_args(argv)
    _install_sigint_handler()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    if not modules or not ranks:
        log("[ERROR] No modules or ranks provided.")
        return 2

    # Auto-detect helpers
    if args.parent_checkpoint is None:
        args.parent_checkpoint = auto_detect_parent_checkpoint(args.model, args.dataset)
        if args.parent_checkpoint:
            log(f"[AUTO] parent_checkpoint = {args.parent_checkpoint}")
        else:
            log("[WARN] Could not auto-detect parent_checkpoint — "
                "each rank will train its baseline from scratch (slower).")

    if args.cached_batches_dir is None:
        args.cached_batches_dir = auto_detect_cached_batches(args.model, args.dataset)
        if args.cached_batches_dir:
            log(f"[AUTO] cached_batches_dir = {args.cached_batches_dir}")

    # VRAM fraction per process
    n_par = max(1, int(args.parallel_ranks))
    if args.max_vram_fraction is not None:
        vram_frac = args.max_vram_fraction
    elif n_par > 1:
        vram_frac = max(0.20, 0.70 / n_par)
        log(f"[VRAM] {n_par} parallel ranks -> {vram_frac:.2f} fraction each")
    else:
        vram_frac = None

    log("=" * 72)
    log("RANK SWEEP RUNNER")
    log(f"  modules         : {modules}")
    log(f"  ranks           : {ranks}")
    log(f"  model           : {args.model}  (family={args.model_family})")
    log(f"  dataset         : {args.dataset}")
    log(f"  parallel_ranks  : {n_par}")
    log(f"  max_vram_frac   : {vram_frac}")
    log(f"  batch_size      : {args.batch_size or 'auto'}")
    log(f"  recovery_steps  : {args.recovery_steps}")
    log(f"  output_base     : {args.output_base}")
    log(f"  fresh           : {args.fresh}")
    log("=" * 72)

    all_rows: List[Dict[str, Any]] = []
    total_t0 = time.time()

    for mod in modules:
        if _interrupted.is_set():
            log("[STOP] Interrupted — skipping remaining modules.")
            break

        log(f"\n{'=' * 72}")
        log(f"Module: {mod}  |  {len(ranks)} ranks  |  parallel={n_par}")
        log(f"{'=' * 72}")

        module_rows: List[Dict[str, Any]] = []
        mod_t0 = time.time()

        if n_par <= 1:
            for rank in ranks:
                if _interrupted.is_set():
                    log(f"[{mod} | r{rank}] skipped (interrupted)")
                    break
                module_rows.append(run_single_rank(mod, rank, args, vram_frac))
        else:
            with ThreadPoolExecutor(max_workers=n_par) as ex:
                futures: Dict[Any, int] = {}
                for rank in ranks:
                    if _interrupted.is_set():
                        break
                    fut = ex.submit(run_single_rank, mod, rank, args, vram_frac)
                    futures[fut] = rank
                for fut in as_completed(futures):
                    rank = futures[fut]
                    try:
                        module_rows.append(fut.result())
                    except Exception as e:
                        log(f"[{mod} | r{rank}] future exception: {e}")
                        module_rows.append(
                            _error_row(mod, rank, f"FUTURE_EXC:{e}", 0.0))

        mod_elapsed = time.time() - mod_t0
        log(f"[{mod}] Module complete in {mod_elapsed:.1f}s "
            f"({len(module_rows)}/{len(ranks)} ranks)")

        write_rank_curve(mod, module_rows, args.output_base)
        print_rank_table(mod, module_rows)
        all_rows.extend(module_rows)

    if all_rows:
        write_combined_curve(all_rows, args.output_base)
        print_summary(all_rows)

    total_elapsed = time.time() - total_t0
    log(f"\n[DONE] Total time: {total_elapsed:.1f}s "
        f"({total_elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Multi-Seed Statistical Rigor Runner & Aggregator.

Runs recovery fine-tuning (50 steps) across 3 deterministic random seeds [42, 123, 456]
for key candidate compressors and attention architectures on DistilBERT and RoBERTa (SST-2).

Produces:
  - results/06_multiseed/multiseed_rigor_{model}/{module}/seed_{seed}/summary.csv
  - results/06_multiseed/multiseed_rigor_summary.csv (Aggregated mean ± std)
  - results/06_multiseed/multiseed_rigor_report.md (Publication-ready markdown table)
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("MultiSeedRigor")

# Root directory of the repository
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Default target modules
KEY_COMPRESSORS = [
    "adasvd_compressor",
    "a3_compressor",
    "kronecker_compressor",
    "mlorq_compressor",
    "rpca_compressor",
    "direct_svd",
    "saes_svd_v2",
    "salr",
    "oats_compressor",
    "cure",
    "admm_compressor",
]

KEY_ATTENTIONS = [
    "nystrom",
    "flash3",
    "chela_slay",
    "nexus",
    "moba",
    "aegis_v2",
    "spectron",
    "performer",
    "rala",
    "cosformer",
    "hgrn2_layer",
]

MODEL_CONFIGS = {
    "distilbert": {
        "model_path": "models/distilbert_sst2",
        "family": "bert",
        "batch_size": 32,
    },
    "roberta": {
        "model_path": "models/roberta_sst2",
        "family": "roberta",
        "batch_size": 16,
    },
}

DEFAULT_SEEDS = [42, 123, 456]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed statistical rigor evaluation on DistilBERT & RoBERTa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="distilbert,roberta",
        help="Comma-separated list of models to evaluate ('distilbert', 'roberta')",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="all",
        help="Comma-separated module list, or 'all', 'compressors', 'attentions'",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated deterministic seeds",
    )
    parser.add_argument(
        "--recovery_steps",
        type=int,
        default=50,
        help="Number of post-injection recovery fine-tuning steps",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating point precision for forward/backward",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/06_multiseed",
        help="Root directory for multi-seed outputs",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing results into summary CSV and markdown report without running",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print execution matrix and skip/todo counts without launching training",
    )
    return parser.parse_args()


def is_run_completed(summary_csv_path: Path) -> tuple[bool, dict | None]:
    """Check if a run has already completed with a valid PASS status."""
    if not summary_csv_path.exists():
        return False, None
    try:
        with open(summary_csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                return False, None
            status = row.get("status", "")
            if "PASS" in status:
                return True, row
            return False, row
    except Exception as e:
        LOG.warning("Failed to parse %s: %s", summary_csv_path, e)
        return False, None


def run_single_module_seed(
    model_key: str,
    module_name: str,
    seed: int,
    args,
    output_dir: Path,
) -> bool:
    """Execute run_quality_sweep.py for a specific model, module, and seed."""
    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["model_path"]
    model_family = cfg["family"]
    batch_size = cfg["batch_size"]

    cached_batches_dir = Path(args.output_root) / f"batch_cache_{model_key}_sst2"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "neuralops" / "benchmarks" / "run_quality_sweep.py"),
        "--model",
        model_path,
        "--model_family",
        model_family,
        "--dataset",
        "sst2",
        "--output_dir",
        str(output_dir),
        "--modules",
        module_name,
        "--batch_size",
        str(batch_size),
        "--recovery_steps",
        str(args.recovery_steps),
        "--precision",
        args.precision,
        "--seed",
        str(seed),
        "--cached_batches_dir",
        str(cached_batches_dir),
        "--defer_export",
    ]

    LOG.info(">>> Launching [%s | %s | Seed %d]", model_key, module_name, seed)
    LOG.info("Command: %s", " ".join(cmd))
    t0 = time.time()

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if proc.returncode == 0:
        summary_csv = output_dir / "summary.csv"
        ok, row = is_run_completed(summary_csv)
        if ok:
            acc = row.get("accuracy_pct", "N/A")
            f1 = row.get("f1_score", "N/A")
            LOG.info(
                "SUCCESS [%s | %s | Seed %d]: Acc=%s%%, F1=%s (Elapsed: %.1fs)",
                model_key,
                module_name,
                seed,
                acc,
                f1,
                elapsed,
            )
            return True
        else:
            LOG.error(
                "Run exited with code 0 but status is not PASS: %s",
                row.get("status") if row else "missing",
            )
            return False
    else:
        LOG.error(
            "FAILED [%s | %s | Seed %d] with returncode %d (Elapsed: %.1fs)",
            model_key,
            module_name,
            seed,
            proc.returncode,
            elapsed,
        )
        return False


def aggregate_results(
    models: list[str],
    all_modules: list[str],
    seeds: list[int],
    output_root: Path,
) -> None:
    """Aggregate all multi-seed runs into summary CSV and markdown report."""
    summary_rows = []

    for model_key in models:
        for mod in all_modules:
            accs = []
            f1s = []
            vrams = []
            times = []
            comp_rates = []
            recon_errors = []
            statuses = []

            for s in seeds:
                seed_csv = (
                    output_root
                    / f"multiseed_rigor_{model_key}"
                    / mod
                    / f"seed_{s}"
                    / "summary.csv"
                )
                ok, row = is_run_completed(seed_csv)
                if ok and row:
                    try:
                        accs.append(float(row.get("accuracy_pct", 0.0)))
                        f1s.append(float(row.get("f1_score", 0.0)))
                        vram_val = float(row.get("peak_vram_mb", 0.0) or 0.0)
                        if vram_val > 0:
                            vrams.append(vram_val)
                        t_val = float(row.get("elapsed_s", 0.0) or 0.0)
                        if t_val > 0:
                            times.append(t_val)
                        crate = row.get("compression_rate", "")
                        if crate and crate != "N/A":
                            comp_rates.append(float(crate))
                        rec = row.get("avg_reconstruction_error", "")
                        if rec and rec != "N/A":
                            recon_errors.append(float(rec))
                        statuses.append(row.get("status", "PASS"))
                    except Exception as parse_err:
                        LOG.debug("Error parsing %s: %s", seed_csv, parse_err)

            num_runs = len(accs)
            if num_runs == 0:
                continue

            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if num_runs > 1 else 0.0
            mean_f1 = statistics.mean(f1s)
            std_f1 = statistics.stdev(f1s) if num_runs > 1 else 0.0
            mean_vram = statistics.mean(vrams) if vrams else 0.0
            mean_time = statistics.mean(times) if times else 0.0
            mean_crate = statistics.mean(comp_rates) if comp_rates else 1.0
            mean_recon = statistics.mean(recon_errors) if recon_errors else None

            cat = "compressor" if mod in KEY_COMPRESSORS else "attention"

            summary_rows.append(
                {
                    "model": model_key,
                    "module": mod,
                    "category": cat,
                    "completed_seeds": f"{num_runs}/{len(seeds)}",
                    "seeds": ",".join(str(s) for s in seeds[:num_runs]),
                    "mean_acc": round(mean_acc, 2),
                    "std_acc": round(std_acc, 2),
                    "mean_f1": round(mean_f1, 4),
                    "std_f1": round(std_f1, 4),
                    "mean_vram_mb": round(mean_vram, 1),
                    "mean_time_s": round(mean_time, 1),
                    "compression_rate": f"{mean_crate:.2f}x",
                    "recon_error": (
                        f"{mean_recon:.4f}" if mean_recon is not None else "—"
                    ),
                    "status": (
                        "PASS" if all("PASS" in st for st in statuses) else "PARTIAL"
                    ),
                }
            )

    # Save summary CSV
    out_csv = output_root / "multiseed_rigor_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        LOG.info(
            "Saved aggregated summary CSV to %s (%d modules)",
            out_csv,
            len(summary_rows),
        )

    # Save Markdown report
    out_md = output_root / "multiseed_rigor_report.md"
    md_lines = [
        "# Multi-Seed Statistical Rigor Evaluation Report",
        "",
        "- **Evaluation Dataset**: SST-2 validation set (N=872)",
        f"- **Seeds**: `{seeds}`",
        "- **Recovery Fine-Tuning**: 50 steps (AdamW, FP16)",
        f"- **Generated Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Summary Table (Mean ± Std Across Deterministic Seeds)",
        "",
        "| Model | Module | Category | Completed Seeds | Accuracy (%) | F1 Score | Compression | Recon Error | Peak VRAM | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in summary_rows:
        acc_str = (
            f"**{r['mean_acc']:.2f} ± {r['std_acc']:.2f}%**"
            if r["completed_seeds"] == f"{len(seeds)}/{len(seeds)}"
            else f"{r['mean_acc']:.2f}%"
        )
        f1_str = (
            f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f}"
            if r["completed_seeds"] == f"{len(seeds)}/{len(seeds)}"
            else f"{r['mean_f1']:.4f}"
        )
        badge = f"`{r['status']}`" if r["status"] == "PASS" else f"**`{r['status']}`**"
        md_lines.append(
            f"| `{r['model']}` | `{r['module']}` | {r['category']} | {r['completed_seeds']} | {acc_str} | {f1_str} | {r['compression_rate']} | {r['recon_error']} | {r['mean_vram_mb']} MB | {badge} |"
        )

    md_lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    LOG.info("Saved Markdown report to %s", out_md)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.modules == "all":
        target_modules = KEY_COMPRESSORS + KEY_ATTENTIONS
    elif args.modules == "compressors":
        target_modules = KEY_COMPRESSORS
    elif args.modules == "attentions":
        target_modules = KEY_ATTENTIONS
    else:
        target_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        LOG.info("Running aggregation only...")
        aggregate_results(models, target_modules, seeds, output_root)
        return

    # Scan and plan runs
    total_runs = len(models) * len(target_modules) * len(seeds)
    pending_runs = []
    completed_runs = []

    for model_key in models:
        for mod in target_modules:
            for s in seeds:
                seed_dir = (
                    output_root / f"multiseed_rigor_{model_key}" / mod / f"seed_{s}"
                )
                summary_csv = seed_dir / "summary.csv"
                completed, _ = is_run_completed(summary_csv)
                if completed:
                    completed_runs.append((model_key, mod, s))
                else:
                    pending_runs.append((model_key, mod, s, seed_dir))

    LOG.info("=" * 80)
    LOG.info("MULTI-SEED STATISTICAL RIGOR SWEEP PLAN")
    LOG.info("  Models:          %s", models)
    LOG.info("  Modules:         %d target modules", len(target_modules))
    LOG.info("  Seeds:           %s", seeds)
    LOG.info("  Total Planned:   %d runs", total_runs)
    LOG.info("  Already PASS:    %d runs", len(completed_runs))
    LOG.info("  Pending / To Do: %d runs", len(pending_runs))
    LOG.info("=" * 80)

    if args.dry_run:
        LOG.info("[DRY RUN] Pending runs to be executed:")
        for m_key, mod, s, _ in pending_runs:
            LOG.info("  - [%s] Module: %-22s Seed: %d", m_key, mod, s)
        LOG.info("[DRY RUN COMPLETE] No sweeps executed.")
        return

    # Execute pending runs sequentially
    success_count = 0
    failure_count = 0

    for i, (m_key, mod, s, seed_dir) in enumerate(pending_runs, 1):
        LOG.info(
            "Progress: [%d / %d pending] Starting %s / %s (seed %d)...",
            i,
            len(pending_runs),
            m_key,
            mod,
            s,
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        ok = run_single_module_seed(m_key, mod, s, args, seed_dir)
        if ok:
            success_count += 1
        else:
            failure_count += 1

        # Periodic intermediate aggregation every 5 runs
        if i % 5 == 0 or i == len(pending_runs):
            LOG.info("Updating intermediate aggregation reports...")
            try:
                aggregate_results(models, target_modules, seeds, output_root)
            except Exception as agg_err:
                LOG.warning("Intermediate aggregation failed: %s", agg_err)

    LOG.info("=" * 80)
    LOG.info(
        "MULTI-SEED SWEEP COMPLETE: %d Succeeded, %d Failed",
        success_count,
        failure_count,
    )
    LOG.info("Performing final statistical aggregation...")
    aggregate_results(models, target_modules, seeds, output_root)
    LOG.info("All done!")


if __name__ == "__main__":
    main()

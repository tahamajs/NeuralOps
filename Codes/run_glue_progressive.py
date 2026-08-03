#!/usr/bin/env python3
"""
Batch GLUE experiment runner – tests all registered attentions & compressors.
Uses the new unified run.py script under the hood.

Usage:
    python batch_runner.py --epochs 1 --max_train_samples 500
    python batch_runner.py --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.getcwd())
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODELS = ["distilbert-base-uncased"]
DATASETS = ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "mnli"]

# Populate registries – import all modules to ensure decorators are executed

ALL_ATTENTIONS = sorted([k for k in ATTENTION_REGISTRY.keys() if k != "full"])
ALL_COMPRESSORS = sorted([k for k in COMPRESSOR_REGISTRY.keys()])

SEEDS = [42]
DEFAULT_MAX_TRAIN_SAMPLES = 1000
DEFAULT_NUM_EPOCHS = 1
DEFAULT_SAVE_VISUALS = True
PLOT_EVERY_N_RUNS = 10

BASE_DIR = Path("./experiments_batch")
BASE_DIR.mkdir(exist_ok=True)
MASTER_CSV = BASE_DIR / "all_results.csv"

# Skip known broken modules (optional)
SKIP_ATTENTIONS = {
    "compressed_attention_layer",
    "compressed_attention_layer_energy",
    "lola_attn",
    "asvd",
    "bam_umoe",
    "cluster",
    "fusion_attn",
    "mixattention_moska",
    "nha_compressor",
    "rala_compressor",
    "jamba_info",
    "optimized_hybrid_attention",
    "spectra_v2",
    "template_compressor",
    "template_attention",
    "adaptive_compressor",
    "tt_compressor",
    "aegis_v5",
    "xattention",
    "progressive_token_pruning",
    "eat",
    "phantom",
    "regla",
    "compression_unified_rope",
    "simple_ssm",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_or_create_master_csv():
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    else:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "base_model",
                "dataset",
                "exp_type",
                "name",
                "method",
                "seed",
                "accuracy",
                "loss",
                "duration_seconds",
                "num_parameters",
                "start_time",
                "end_time",
            ]
        )
    return df


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def save_cumulative_plot(df, output_path, since_run=None):
    df = df.copy()
    df["accuracy"] = safe_numeric(df["accuracy"])
    df = df.dropna(subset=["accuracy"])
    if df.empty:
        return
    pivot = df[df["exp_type"] == "attention"].pivot_table(
        index="name", columns="dataset", values="accuracy", aggfunc="mean"
    )
    if not pivot.empty:
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot, annot=True, cmap="viridis", vmin=0, vmax=1, fmt=".3f")
        title = "Mean accuracy: Attention × Dataset"
        if since_run:
            title += f" (after {since_run} runs)"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"📊 Saved cumulative plot: {output_path}")


def run_single_experiment(base_model, dataset, exp_type, name, method, seed, args):
    """
    Launch a single experiment using the unified run.py script.
    Returns a dict with the result row.
    """
    if exp_type == "attention":
        exp_name = (
            f"attn_{name}_{method}_{dataset}_{base_model.replace('/', '_')}_s{seed}"
        )
        cmd = [
            sys.executable,
            "run.py",
            "--model",
            base_model,
            "--attn_type",
            name,
            "--compressor",
            "none",
            "--method",
            method,
            "--task",
            dataset,
            "--seed",
            str(seed),
            "--output_dir",
            str(BASE_DIR),
            "--exp_name",
            exp_name,
            "--epochs",
            str(args.num_epochs),
            "--batch_size",
            "16",
            "--rank",
            "32",
        ]
    else:
        exp_name = (
            f"comp_{name}_{method}_{dataset}_{base_model.replace('/', '_')}_s{seed}"
        )
        cmd = [
            sys.executable,
            "run.py",
            "--model",
            base_model,
            "--attn_type",
            "full",
            "--compressor",
            name,
            "--method",
            method,
            "--task",
            dataset,
            "--seed",
            str(seed),
            "--output_dir",
            str(BASE_DIR),
            "--exp_name",
            exp_name,
            "--epochs",
            str(args.num_epochs),
            "--batch_size",
            "16",
        ]

    if args.max_train_samples > 0:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if args.save_visuals:
        cmd.append("--save_visuals")
    if args.use_ema:
        cmd.append("--use_ema")
    if args.gradient_clip_norm is not None:
        cmd += ["--gradient_clip_norm", str(args.gradient_clip_norm)]
    if args.attention_heatmap_steps > 0:
        cmd += ["--attention_heatmap_steps", str(args.attention_heatmap_steps)]
    if args.svd_analysis_steps > 0:
        cmd += ["--svd_analysis_steps", str(args.svd_analysis_steps)]
    if args.log_grad_norm:
        cmd.append("--log_grad_norm")
    if args.activation_stats:
        cmd.append("--activation_stats")

    output_dir = BASE_DIR / dataset / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"RUNNING: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()

    log_file = output_dir / "run.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()

    duration = time.time() - start_time

    metrics_file = output_dir / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        accuracy = metrics.get("final_accuracy", "NA")
        loss = metrics.get("final_loss", "NA")
        n_params = metrics.get("num_parameters", "NA")
        start_iso = metrics.get("start_time", "NA")
        end_iso = metrics.get("end_time", "NA")
    else:
        accuracy = loss = n_params = start_iso = end_iso = "NA"

    return {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "dataset": dataset,
        "exp_type": exp_type,
        "name": name,
        "method": method,
        "seed": seed,
        "accuracy": accuracy,
        "loss": loss,
        "duration_seconds": round(duration, 2),
        "num_parameters": n_params,
        "start_time": start_iso,
        "end_time": end_iso,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES
    )
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument(
        "--save_visuals", action="store_true", default=DEFAULT_SAVE_VISUALS
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--attention_heatmap_steps", type=int, default=0)
    parser.add_argument("--svd_analysis_steps", type=int, default=0)
    parser.add_argument("--log_grad_norm", action="store_true")
    parser.add_argument("--activation_stats", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit_attentions", type=int, default=None)
    parser.add_argument("--limit_compressors", type=int, default=None)
    args = parser.parse_args()

    results_df = load_or_create_master_csv()
    completed_runs = len(results_df)

    # Filter out broken modules
    attentions = [a for a in ALL_ATTENTIONS if a not in SKIP_ATTENTIONS]
    compressors = [c for c in ALL_COMPRESSORS if c not in SKIP_ATTENTIONS]
    if args.limit_attentions:
        attentions = attentions[: args.limit_attentions]
    if args.limit_compressors:
        compressors = compressors[: args.limit_compressors]

    methods = ["baseline"]  # expand as needed

    total = (
        len(BASE_MODELS)
        * len(DATASETS)
        * (len(attentions) * len(methods) + len(compressors) * len(methods))
        * len(SEEDS)
    )
    print(f"Total experiments: {total}")
    print(f"Already completed: {completed_runs}")
    print(f"Resume mode: {args.resume}")

    run_counter = 0
    for base_model in BASE_MODELS:
        for dataset in DATASETS:
            # Attentions
            for attn in attentions:
                for method in methods:
                    for seed in SEEDS:
                        if args.resume:
                            mask = (
                                (results_df["base_model"] == base_model)
                                & (results_df["dataset"] == dataset)
                                & (results_df["exp_type"] == "attention")
                                & (results_df["name"] == attn)
                                & (results_df["method"] == method)
                                & (results_df["seed"] == seed)
                            )
                            if not results_df[mask].empty:
                                continue
                        row = run_single_experiment(
                            base_model, dataset, "attention", attn, method, seed, args
                        )
                        results_df = pd.concat(
                            [results_df, pd.DataFrame([row])], ignore_index=True
                        )
                        completed_runs += 1
                        run_counter += 1
                        results_df.to_csv(MASTER_CSV, index=False)
                        print(f"✅ {completed_runs}/{total} runs completed")
                        if run_counter % PLOT_EVERY_N_RUNS == 0:
                            save_cumulative_plot(
                                results_df,
                                BASE_DIR / f"cumulative_{completed_runs}.png",
                                completed_runs,
                            )

            # Compressors
            for comp in compressors:
                for method in methods:
                    for seed in SEEDS:
                        if args.resume:
                            mask = (
                                (results_df["base_model"] == base_model)
                                & (results_df["dataset"] == dataset)
                                & (results_df["exp_type"] == "compressor")
                                & (results_df["name"] == comp)
                                & (results_df["method"] == method)
                                & (results_df["seed"] == seed)
                            )
                            if not results_df[mask].empty:
                                continue
                        row = run_single_experiment(
                            base_model, dataset, "compressor", comp, method, seed, args
                        )
                        results_df = pd.concat(
                            [results_df, pd.DataFrame([row])], ignore_index=True
                        )
                        completed_runs += 1
                        run_counter += 1
                        results_df.to_csv(MASTER_CSV, index=False)
                        print(f"✅ {completed_runs}/{total} runs completed")
                        if run_counter % PLOT_EVERY_N_RUNS == 0:
                            save_cumulative_plot(
                                results_df,
                                BASE_DIR / f"cumulative_{completed_runs}.png",
                                completed_runs,
                            )

    # Final plots
    save_cumulative_plot(results_df, BASE_DIR / "final_cumulative_plot.png")
    pivot = results_df.pivot_table(
        index=["base_model", "exp_type", "name"],
        columns="dataset",
        values="accuracy",
        aggfunc="mean",
    )
    pivot.to_csv(BASE_DIR / "pivot_summary.csv")
    print("\n🎉 ALL EXPERIMENTS COMPLETED")
    print(f"Master CSV: {MASTER_CSV}")
    print(f"Pivot summary: {BASE_DIR / 'pivot_summary.csv'}")


if __name__ == "__main__":
    main()

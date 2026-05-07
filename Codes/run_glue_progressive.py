#!/usr/bin/env python3
"""
GLUE experiment runner – test attentions and compressors separately.
Supports all callbacks: attention_heatmap_steps, svd_analysis_steps, log_grad_norm, activation_stats.
"""

import subprocess
import sys
import os
import time
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.getcwd())
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

# ---------- Configuration ----------
BASE_MODELS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
DATASETS = ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "mnli"]

ALL_ATTENTIONS = [k for k in ATTENTION_REGISTRY.keys() if k != "full"]
ALL_COMPRESSORS = [k for k in COMPRESSOR_REGISTRY.keys()]

SEEDS = [42]

DEFAULT_MAX_TRAIN_SAMPLES = 1000
DEFAULT_NUM_EPOCHS = 1
DEFAULT_SAVE_VISUALS = True
PLOT_EVERY_N_RUNS = 10

BASE_DIR = Path("./experiments_full_separate")
BASE_DIR.mkdir(exist_ok=True)
MASTER_CSV = BASE_DIR / "all_results.csv"

if MASTER_CSV.exists():
    results_df = pd.read_csv(MASTER_CSV)
else:
    results_df = pd.DataFrame(columns=[
        "timestamp", "base_model", "dataset", "exp_type", "name",
        "method", "seed", "accuracy", "loss", "duration_seconds",
        "num_parameters", "start_time", "end_time"
    ])
completed_runs = len(results_df)

def run_experiment(base_model, dataset, exp_type, name, method, seed, args):
    if exp_type == "attention":
        exp_name = f"attn_{name}_method_{method}_{dataset}_{base_model.replace('/', '_')}_seed{seed}"
        output_dir = BASE_DIR / dataset / exp_name
        cmd = [
            sys.executable, "main.py",
            "--model", base_model,
            "--attn_type", name,
            "--compressor", "none",
            "--method", method,
            "--task", dataset,
            "--seed", str(seed),
            "--output_dir", str(BASE_DIR / dataset),
            "--exp_name", exp_name,
            "--num_train_epochs", str(args.num_epochs),
        ]
    else:
        exp_name = f"comp_{name}_method_{method}_{dataset}_{base_model.replace('/', '_')}_seed{seed}"
        output_dir = BASE_DIR / dataset / exp_name
        cmd = [
            sys.executable, "main.py",
            "--model", base_model,
            "--attn_type", "full",
            "--compressor", name,
            "--method", method,
            "--task", dataset,
            "--seed", str(seed),
            "--output_dir", str(BASE_DIR / dataset),
            "--exp_name", exp_name,
            "--num_train_epochs", str(args.num_epochs),
        ]

    if args.max_train_samples > 0:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if args.save_visuals:
        cmd.append("--save_visuals")
    if args.use_ema:
        cmd.append("--use_ema")
    if args.gradient_clip_norm is not None:
        cmd += ["--gradient_clip_norm", str(args.gradient_clip_norm)]
    # --- Forward all analysis callbacks ---
    if args.attention_heatmap_steps > 0:
        cmd += ["--attention_heatmap_steps", str(args.attention_heatmap_steps)]
    if args.svd_analysis_steps > 0:
        cmd += ["--svd_analysis_steps", str(args.svd_analysis_steps)]
    if args.log_grad_norm:
        cmd.append("--log_grad_norm")
    if args.activation_stats:
        cmd.append("--activation_stats")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"RUNNING: {exp_name}")
    print(f"Model: {base_model}, dataset: {dataset}, {exp_type}: {name}, method: {method}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    with open(output_dir / "run.log", "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    end_time = time.time()
    duration = end_time - start_time

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

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def save_plot(df, exp_type, output_path):
    df = df.copy()
    df["accuracy"] = safe_numeric(df["accuracy"])
    df = df.dropna(subset=["accuracy"])
    if df.empty:
        return
    plt.figure(figsize=(14,6))
    sns.barplot(data=df, x="dataset", y="accuracy", hue="name", ci="sd")
    plt.title(f"{exp_type.capitalize()} performance")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved plot: {output_path}")

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
        plt.figure(figsize=(14,10))
        sns.heatmap(pivot, annot=True, cmap="viridis", vmin=0, vmax=1, fmt=".3f")
        title = "Mean accuracy: Attention × Dataset"
        if since_run:
            title += f" (after {since_run} runs)"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"📊 Saved cumulative plot: {output_path}")

def main():
    global results_df, completed_runs

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--save_visuals", action="store_true", default=DEFAULT_SAVE_VISUALS)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    # Analysis callbacks
    parser.add_argument("--attention_heatmap_steps", type=int, default=0)
    parser.add_argument("--svd_analysis_steps", type=int, default=0)
    parser.add_argument("--log_grad_norm", action="store_true")
    parser.add_argument("--activation_stats", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit_attentions", type=int, default=None)
    parser.add_argument("--limit_compressors", type=int, default=None)
    args = parser.parse_args()

    attentions = ALL_ATTENTIONS[:args.limit_attentions] if args.limit_attentions else ALL_ATTENTIONS
    compressors = ALL_COMPRESSORS[:args.limit_compressors] if args.limit_compressors else ALL_COMPRESSORS
    methods = ["baseline", "lora"]

    total = len(BASE_MODELS) * len(DATASETS) * (
        len(attentions) * len(methods) + len(compressors) * len(methods)
    ) * len(SEEDS)
    print(f"Total experiments: {total}")
    print(f"Completed already: {completed_runs}")
    print(f"Resume mode: {args.resume}")
    print(f"Attentions to test: {len(attentions)}")
    print(f"Compressors to test: {len(compressors)}\n")

    run_counter = 0
    for base_model in BASE_MODELS:
        for dataset in DATASETS:
            # ---- Attentions ----
            for attn in attentions:
                for method in methods:
                    for seed in SEEDS:
                        if args.resume and not results_df[
                            (results_df["base_model"] == base_model) &
                            (results_df["dataset"] == dataset) &
                            (results_df["exp_type"] == "attention") &
                            (results_df["name"] == attn) &
                            (results_df["method"] == method) &
                            (results_df["seed"] == seed)
                        ].empty:
                            continue
                        new_row = run_experiment(base_model, dataset, "attention", attn, method, seed, args)
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        completed_runs += 1
                        run_counter += 1
                        results_df.to_csv(MASTER_CSV, index=False)
                        print(f"✅ Completed {completed_runs}/{total}")
                        if run_counter % PLOT_EVERY_N_RUNS == 0:
                            cum_plot = BASE_DIR / f"cumulative_after_{completed_runs}_runs.png"
                            save_cumulative_plot(results_df, cum_plot, completed_runs)
            # ---- Compressors ----
            for comp in compressors:
                for method in methods:
                    for seed in SEEDS:
                        if args.resume and not results_df[
                            (results_df["base_model"] == base_model) &
                            (results_df["dataset"] == dataset) &
                            (results_df["exp_type"] == "compressor") &
                            (results_df["name"] == comp) &
                            (results_df["method"] == method) &
                            (results_df["seed"] == seed)
                        ].empty:
                            continue
                        new_row = run_experiment(base_model, dataset, "compressor", comp, method, seed, args)
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        completed_runs += 1
                        run_counter += 1
                        results_df.to_csv(MASTER_CSV, index=False)
                        print(f"✅ Completed {completed_runs}/{total}")
                        if run_counter % PLOT_EVERY_N_RUNS == 0:
                            cum_plot = BASE_DIR / f"cumulative_after_{completed_runs}_runs.png"
                            save_cumulative_plot(results_df, cum_plot, completed_runs)

            # After dataset, save per‑dataset plots
            ds_df = results_df[(results_df["base_model"] == base_model) & (results_df["dataset"] == dataset)]
            if not ds_df.empty:
                attn_plot = BASE_DIR / f"summary_{base_model}_{dataset}_attentions.png"
                comp_plot = BASE_DIR / f"summary_{base_model}_{dataset}_compressors.png"
                save_plot(ds_df[ds_df["exp_type"] == "attention"], "attention", attn_plot)
                save_plot(ds_df[ds_df["exp_type"] == "compressor"], "compressor", comp_plot)

    save_cumulative_plot(results_df, BASE_DIR / "final_cumulative_plot.png")
    print("\n🎉 ALL EXPERIMENTS FINISHED")
    print(f"Master CSV: {MASTER_CSV}")
    print(f"All experiment folders under: {BASE_DIR}")

if __name__ == "__main__":
    main()
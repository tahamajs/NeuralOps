#!/usr/bin/env python3
"""
batch_runner.py – Efficiently run all combinations of models, tasks,
attentions, compressors, methods, and seeds.
Uses the unified run.py script, supports resumption, and produces
cumulative plots automatically.

Usage:
    python batch_runner.py --epochs 1 --max_samples 500
    python batch_runner.py --resume
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure framework is on the path
sys.path.insert(0, os.getcwd())
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.training.methods import available_methods

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODELS = ["distilbert-base-uncased"]
DATASETS = ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "mnli"]
SEEDS = [42]
DEFAULT_EPOCHS = 1
DEFAULT_MAX_SAMPLES = 1000
PARALLEL_WORKERS = 1          # increase for multi‑GPU or multi‑CPU

BASE_DIR = Path("./experiments_batch")
BASE_DIR.mkdir(exist_ok=True)
MASTER_CSV = BASE_DIR / "all_results.csv"

# Populate registries
import framework.attentions
# import framework.compressors   # if you have a separate compressors package

ALL_ATTENTIONS = sorted([k for k in ATTENTION_REGISTRY.keys() if k != "full"])
ALL_COMPRESSORS = sorted([k for k in COMPRESSOR_REGISTRY.keys()])
ALL_METHODS = available_methods() if available_methods() else ["baseline"]

# Skip known broken modules (optional)
SKIP = set([
    "compressed_attention_layer", "compressed_attention_layer_energy",
    "lola_attn", "asvd", "bam_umoe", "cluster", "fusion_attn",
    "mixattention_moska", "nha_compressor", "rala_compressor",
    "jamba_info", "optimized_hybrid_attention", "spectra_v2",
    "template_compressor", "template_attention", "adaptive_compressor",
    "tt_compressor", "aegis_v5", "xattention", "progressive_token_pruning",
    "eat", "phantom", "regla", "compression_unified_rope", "simple_ssm"
])

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_results_df():
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV)
    return pd.DataFrame(columns=[
        "timestamp", "base_model", "dataset", "exp_type", "name",
        "method", "seed", "accuracy", "loss", "duration_seconds",
        "num_parameters", "start_time", "end_time"
    ])

def save_cumulative_plot(df, path, title="Attention × Dataset"):
    df = df.copy()
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
    df = df.dropna(subset=["accuracy"])
    if df.empty:
        return
    pivot = df[df["exp_type"] == "attention"].pivot_table(
        index="name", columns="dataset", values="accuracy", aggfunc="mean"
    )
    if pivot.empty:
        return
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, cmap="viridis", vmin=0, vmax=1, fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def run_one(cmd, log_file):
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            f.write(line)
        proc.wait()
    return proc.returncode

def build_cmd(base_model, dataset, exp_type, name, method, seed, args, output_dir):
    if exp_type == "attention":
        exp_name = f"attn_{name}_{method}_{dataset}_{base_model.replace('/', '_')}_s{seed}"
        return [
            sys.executable, "run.py",
            "--model", base_model,
            "--attn_type", name,
            "--compressor", "none",
            "--method", method,
            "--task", dataset,
            "--seed", str(seed),
            "--output_dir", str(output_dir),
            "--exp_name", exp_name,
            "--epochs", str(args.epochs),
            "--batch_size", "16",
            "--rank", "32",
        ], exp_name
    else:
        exp_name = f"comp_{name}_{method}_{dataset}_{base_model.replace('/', '_')}_s{seed}"
        return [
            sys.executable, "run.py",
            "--model", base_model,
            "--attn_type", "full",
            "--compressor", name,
            "--method", method,
            "--task", dataset,
            "--seed", str(seed),
            "--output_dir", str(output_dir),
            "--exp_name", exp_name,
            "--epochs", str(args.epochs),
            "--batch_size", "16",
        ], exp_name

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", type=int, default=PARALLEL_WORKERS)
    args = parser.parse_args()

    results_df = load_results_df()
    finished = len(results_df)

    attentions = [a for a in ALL_ATTENTIONS if a not in SKIP]
    compressors = [c for c in ALL_COMPRESSORS if c not in SKIP]

    # Build list of all experiments
    experiments = []
    for model in BASE_MODELS:
        for dataset in DATASETS:
            for attn in attentions:
                for method in ALL_METHODS:
                    for seed in SEEDS:
                        key = (model, dataset, "attention", attn, method, seed)
                        experiments.append(key)
            for comp in compressors:
                for method in ALL_METHODS:
                    for seed in SEEDS:
                        key = (model, dataset, "compressor", comp, method, seed)
                        experiments.append(key)

    total = len(experiments)
    print(f"Total experiments: {total} (finished: {finished})")

    # Filter already finished (resume)
    if args.resume:
        finished_set = set()
        for _, row in results_df.iterrows():
            finished_set.add((
                row["base_model"], row["dataset"], row["exp_type"],
                row["name"], row["method"], row["seed"]
            ))
        experiments = [e for e in experiments if e not in finished_set]
        print(f"Resuming: {len(experiments)} remaining out of {total}")

    # Sequential or parallel
    if args.parallel <= 1:
        for (base_model, dataset, exp_type, name, method, seed) in experiments:
            cmd, exp_name = build_cmd(base_model, dataset, exp_type, name, method, seed, args, BASE_DIR)
            output_dir = BASE_DIR / dataset / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "run.log"
            start_time = time.time()
            print(f"Running: {exp_name}")
            ret = run_one(cmd, log_file)
            duration = time.time() - start_time
            metrics_file = output_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                acc = metrics.get("final_accuracy", "NA")
                loss = metrics.get("final_loss", "NA")
                n_params = metrics.get("num_parameters", "NA")
            else:
                acc = loss = n_params = "NA"
            new_row = {
                "timestamp": datetime.now().isoformat(),
                "base_model": base_model, "dataset": dataset,
                "exp_type": exp_type, "name": name,
                "method": method, "seed": seed,
                "accuracy": acc, "loss": loss,
                "duration_seconds": round(duration, 2),
                "num_parameters": n_params,
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(MASTER_CSV, index=False)
            print(f"Completed {len(results_df)}/{total}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for (base_model, dataset, exp_type, name, method, seed) in experiments:
                cmd, exp_name = build_cmd(base_model, dataset, exp_type, name, method, seed, args, BASE_DIR)
                output_dir = BASE_DIR / dataset / exp_name
                output_dir.mkdir(parents=True, exist_ok=True)
                log_file = output_dir / "run.log"
                future = executor.submit(run_one, cmd, log_file)
                futures[future] = (base_model, dataset, exp_type, name, method, seed, output_dir, exp_name, time.time())

            for future in as_completed(futures):
                base_model, dataset, exp_type, name, method, seed, output_dir, exp_name, start_time = futures[future]
                ret = future.result()
                duration = time.time() - start_time
                metrics_file = output_dir / "final_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    acc = metrics.get("final_accuracy", "NA")
                    loss = metrics.get("final_loss", "NA")
                    n_params = metrics.get("num_parameters", "NA")
                else:
                    acc = loss = n_params = "NA"
                new_row = {
                    "timestamp": datetime.now().isoformat(),
                    "base_model": base_model, "dataset": dataset,
                    "exp_type": exp_type, "name": name,
                    "method": method, "seed": seed,
                    "accuracy": acc, "loss": loss,
                    "duration_seconds": round(duration, 2),
                    "num_parameters": n_params,
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                results_df.to_csv(MASTER_CSV, index=False)
                print(f"Finished {exp_name} ({len(results_df)}/{total})")

    # Final summary
    save_cumulative_plot(results_df, BASE_DIR / "final_cumulative.png")
    pivot = results_df.pivot_table(
        index=["base_model", "exp_type", "name"],
        columns="dataset",
        values="accuracy",
        aggfunc="mean"
    )
    pivot.to_csv(BASE_DIR / "pivot_summary.csv")
    print(f"\nAll experiments done. Results in {BASE_DIR}")

if __name__ == "__main__":
    main()
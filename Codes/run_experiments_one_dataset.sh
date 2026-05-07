#!/usr/bin/env python3
"""
Progressive GLUE Experiment Orchestrator
- Runs experiments dataset × attention × compressor × method × seed
- Saves intermediate plots after each dataset and after every N runs
- Appends results to a master CSV
- Calls main.py as a subprocess to avoid state pollution
"""

import subprocess
import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ========== CONFIGURATION ==========
DATASETS = ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "mnli"]
# MNLI has matched/mismatched but we use the main 'mnli' task – code handles it.
ATTENTIONS = ["token_compressed", "aegis_v4", "gqa", "full"]
COMPRESSORS = ["tt", "none"]
METHODS = ["baseline", "lora"]
SEEDS = [42]

# Experiment limits
MAX_TRAIN_SAMPLES = 2000   # 0 = no limit
NUM_EPOCHS = 1
SAVE_VISUALS = True

# Progressive saving
PLOT_EVERY_N_RUNS = 10     # generate cumulative plot after every 10 completed experiments

# Directories
BASE_DIR = Path("./experiments_all_glue")
BASE_DIR.mkdir(exist_ok=True)
MASTER_CSV = BASE_DIR / "all_glue_results.csv"

# Global results dataframe (load existing or new)
if MASTER_CSV.exists():
    results_df = pd.read_csv(MASTER_CSV)
else:
    results_df = pd.DataFrame(columns=[
        "timestamp", "dataset", "attention", "compressor", "method", "seed",
        "accuracy", "loss", "duration_seconds", "num_parameters", "start_time", "end_time"
    ])

# Counter for total completed runs
completed_runs = len(results_df)

# ========== Helper: run a single experiment ==========
def run_experiment(dataset, attention, compressor, method, seed):
    exp_name = f"{attention}_{compressor}_{method}_{dataset}_seed{seed}"
    output_dir = BASE_DIR / dataset / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command line
    cmd = [
        sys.executable, "main.py",
        "--attn_type", attention,
        "--compressor", compressor,
        "--method", method,
        "--task", dataset,
        "--seed", str(seed),
        "--output_dir", str(BASE_DIR / dataset),
        "--exp_name", exp_name,
        "--num_train_epochs", str(NUM_EPOCHS),
    ]
    if MAX_TRAIN_SAMPLES > 0:
        cmd += ["--max_train_samples", str(MAX_TRAIN_SAMPLES)]
    if SAVE_VISUALS:
        cmd += ["--save_visuals"]
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {exp_name} | dataset={dataset}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    # Run main.py and capture output
    with open(output_dir / "run.log", "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Parse final_metrics.json
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
    
    # Append to results dataframe
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "attention": attention,
        "compressor": compressor,
        "method": method,
        "seed": seed,
        "accuracy": accuracy,
        "loss": loss,
        "duration_seconds": round(duration, 2),
        "num_parameters": n_params,
        "start_time": start_iso,
        "end_time": end_iso,
    }
    return new_row

# ========== Plotting functions ==========
def save_dataset_plot(dataset, df_subset, output_path):
    """Create a bar plot for a single dataset (accuracy per attention/method)"""
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_subset, x="attention", y="accuracy", hue="method", ci="sd")
    plt.title(f"Accuracy on {dataset.upper()} (1 epoch, {MAX_TRAIN_SAMPLES} samples)")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved dataset plot: {output_path}")

def save_cumulative_plot(df, output_path, since_run=None):
    """Create a cumulative heatmap (attention vs dataset) from all results so far"""
    if df.empty:
        return
    # Average accuracy per attention and dataset
    pivot = df.pivot_table(index="attention", columns="dataset", values="accuracy", aggfunc="mean")
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, annot=True, cmap="viridis", vmin=0, vmax=1, fmt=".3f")
    title = "Cumulative mean accuracy: Attention × Dataset"
    if since_run is not None:
        title += f" (after {since_run} experiments)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved cumulative plot: {output_path}")

# ========== Main orchestration ==========
def main():
    global results_df, completed_runs
    
    # Ensure CSV header exists
    if not MASTER_CSV.exists():
        results_df.to_csv(MASTER_CSV, index=False)
    
    total_combinations = len(DATASETS) * len(ATTENTIONS) * len(COMPRESSORS) * len(METHODS) * len(SEEDS)
    print(f"Total experiments to run: {total_combinations}")
    print(f"Resuming from {completed_runs} already completed runs.")
    
    run_counter = 0
    for dataset in DATASETS:
        dataset_complete = False
        for attention in ATTENTIONS:
            for compressor in COMPRESSORS:
                for method in METHODS:
                    for seed in SEEDS:
                        # Skip if already run (check by existence in results_df)
                        if not results_df[
                            (results_df["dataset"] == dataset) &
                            (results_df["attention"] == attention) &
                            (results_df["compressor"] == compressor) &
                            (results_df["method"] == method) &
                            (results_df["seed"] == seed)
                        ].empty:
                            print(f"Skipping already completed: {dataset}/{attention}/{compressor}/{method}/{seed}")
                            continue
                        
                        # Run experiment
                        new_row = run_experiment(dataset, attention, compressor, method, seed)
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        completed_runs += 1
                        run_counter += 1
                        
                        # Save after each experiment
                        results_df.to_csv(MASTER_CSV, index=False)
                        print(f"✅ Completed {completed_runs}/{total_combinations} experiments")
                        
                        # After every N runs, save cumulative plot
                        if run_counter % PLOT_EVERY_N_RUNS == 0:
                            cum_plot = BASE_DIR / f"cumulative_after_{completed_runs}_runs.png"
                            save_cumulative_plot(results_df, cum_plot, since_run=completed_runs)
        
        # After finishing all combinations for a dataset, generate dataset-specific plot
        if not dataset_complete:  # just finished this dataset
            dataset_df = results_df[results_df["dataset"] == dataset].copy()
            if not dataset_df.empty:
                # Convert accuracy to numeric
                dataset_df["accuracy"] = pd.to_numeric(dataset_df["accuracy"], errors="coerce")
                dataset_df = dataset_df.dropna(subset=["accuracy"])
                if not dataset_df.empty:
                    plot_path = BASE_DIR / f"summary_{dataset}.png"
                    save_dataset_plot(dataset, dataset_df, plot_path)
        dataset_complete = True
    
    # Final cumulative plot
    save_cumulative_plot(results_df, BASE_DIR / "final_cumulative_plot.png")
    print("\n🎉 ALL EXPERIMENTS FINISHED")
    print(f"Master CSV saved: {MASTER_CSV}")
    print(f"All plots saved in {BASE_DIR}")

if __name__ == "__main__":
    main()
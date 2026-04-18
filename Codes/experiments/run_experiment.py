#!/usr/bin/env python3
"""
Generate CSV reports and comparison plots from experiment results.
Usage:
    python scripts/generate_report.py --results_dir results/run_20250418_143022
    python scripts/generate_report.py --compare_runs run1 run2 run3 --output_dir comparison
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style for publication‑ready plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CSV reports and comparison plots")
    parser.add_argument("--results_dir", type=str, help="Path to a single run folder (e.g., results/run_20250418_143022)")
    parser.add_argument("--compare_runs", nargs="+", help="List of run folders to compare (e.g., run_20250418_143022 run_20250419_091234)")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for CSV and plots")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots, only CSV")
    return parser.parse_args()


def load_run_results(run_path):
    """Load all summary.json files from a run folder."""
    all_results = {}
    run_name = os.path.basename(run_path)
    for task_dir in Path(run_path).iterdir():
        if not task_dir.is_dir():
            continue
        summary_file = task_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                task_results = json.load(f)
            # Add run name and task to each metric entry
            for rank, metrics in task_results.items():
                if "error" in metrics:
                    continue
                metrics["run"] = run_name
                metrics["task"] = task_dir.name
                metrics["rank"] = rank if rank != "None" else "full"
            all_results[task_dir.name] = task_results
    return all_results


def build_dataframe(all_results):
    """Convert nested results into a flat DataFrame."""
    rows = []
    for task, task_results in all_results.items():
        for rank, metrics in task_results.items():
            if "error" in metrics:
                continue
            row = {
                "run": metrics.get("run", "unknown"),
                "task": task,
                "rank": rank,
            }
            # Extract metrics
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    metric_name = key.replace("eval_", "")
                    row[metric_name] = value
                elif key in ["loss", "accuracy", "matthews_correlation", "pearson", "spearman"]:
                    row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy_vs_rank(df, output_dir):
    """Plot accuracy (or primary metric) vs rank for each task."""
    tasks = df["task"].unique()
    for task in tasks:
        df_task = df[df["task"] == task].copy()
        # Convert rank to numeric, handle 'full'
        df_task["rank_num"] = df_task["rank"].apply(lambda x: 10000 if x == "full" else int(x))
        df_task = df_task.sort_values("rank_num")
        
        # Determine primary metric
        if task == "stsb":
            metric = "pearson"
            ylabel = "Pearson Correlation"
        elif task == "cola":
            metric = "matthews_correlation"
            ylabel = "Matthews Correlation"
        else:
            metric = "accuracy"
            ylabel = "Accuracy"
        
        plt.figure(figsize=(10, 6))
        # Plot each run separately if multiple runs
        runs = df_task["run"].unique()
        for run in runs:
            df_run = df_task[df_task["run"] == run]
            ranks = df_run["rank_num"].values
            scores = df_run[metric].values
            # Replace 10000 with 'full' for x-tick label
            labels = [str(r) if r != 10000 else "full" for r in ranks]
            plt.plot(range(len(ranks)), scores, marker='o', linewidth=2, markersize=8, label=run)
            plt.xticks(range(len(ranks)), labels)
        plt.xlabel("Rank")
        plt.ylabel(ylabel)
        plt.title(f"{task.upper()} - Performance vs Compression Rank")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rank_vs_{metric}_{task}.png"), dpi=150)
        plt.close()
    print(f"Accuracy vs rank plots saved to {output_dir}")


def plot_model_comparison(df, output_dir, metric="accuracy"):
    """Bar chart comparing different models (runs) for a fixed rank."""
    # For each task, plot bar chart of metric across runs
    tasks = df["task"].unique()
    # Choose a representative rank (e.g., 128) if present, else best rank
    for task in tasks:
        df_task = df[df["task"] == task]
        # If multiple ranks, pick rank 128 if exists, else full
        if 128 in df_task["rank"].astype(str).values:
            df_task = df_task[df_task["rank"].astype(str) == "128"]
        elif "full" in df_task["rank"].values:
            df_task = df_task[df_task["rank"] == "full"]
        else:
            # pick first rank
            df_task = df_task.iloc[:1]
        
        if df_task.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_task, x="run", y=metric, palette="viridis")
        plt.xlabel("Model / Run")
        plt.ylabel(metric.capitalize())
        plt.title(f"{task.upper()} - Model Comparison ({metric})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"model_comparison_{task}_{metric}.png"), dpi=150)
        plt.close()
    print(f"Model comparison plots saved to {output_dir}")


def plot_training_curves(run_path, output_dir):
    """Extract training loss and eval metrics from trainer logs."""
    # Look for metrics_log.json in the run folder
    log_file = os.path.join(run_path, "metrics_log.json")
    if not os.path.exists(log_file):
        print(f"No metrics_log.json found in {run_path}, skipping training curves.")
        return
    
    records = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return
    
    df_log = pd.DataFrame(records)
    # Plot training loss over steps
    if "loss" in df_log.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df_log["step"], df_log["loss"], label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss - {os.path.basename(run_path)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_loss_{os.path.basename(run_path)}.png"), dpi=150)
        plt.close()
    
    # Plot eval metrics over epoch
    eval_data = df_log[df_log["name"].str.contains("eval_", na=False)]
    if not eval_data.empty:
        plt.figure(figsize=(10, 5))
        for metric in eval_data["name"].unique():
            df_metric = eval_data[eval_data["name"] == metric]
            plt.plot(df_metric["step"], df_metric["value"], marker='o', label=metric)
        plt.xlabel("Step")
        plt.ylabel("Metric Value")
        plt.title(f"Evaluation Metrics - {os.path.basename(run_path)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"eval_metrics_{os.path.basename(run_path)}.png"), dpi=150)
        plt.close()


def generate_csv_summary(df, output_dir):
    """Save DataFrame to CSV."""
    csv_path = os.path.join(output_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV summary saved to {csv_path}")
    
    # Also create per‑task CSV
    for task in df["task"].unique():
        df_task = df[df["task"] == task]
        task_csv = os.path.join(output_dir, f"{task}_results.csv")
        df_task.to_csv(task_csv, index=False)
    return csv_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.results_dir:
        # Single run analysis
        print(f"Analyzing single run: {args.results_dir}")
        all_results = load_run_results(args.results_dir)
        df = build_dataframe(all_results)
        generate_csv_summary(df, args.output_dir)
        if not args.no_plots:
            plot_accuracy_vs_rank(df, args.output_dir)
            plot_training_curves(args.results_dir, args.output_dir)
            # For single run, model comparison not meaningful, but we can still plot per task
            plot_model_comparison(df, args.output_dir)
    
    elif args.compare_runs:
        # Compare multiple runs
        print(f"Comparing runs: {args.compare_runs}")
        all_dfs = []
        for run_name in args.compare_runs:
            # run_name could be full path or just folder name under results/
            run_path = run_name if os.path.exists(run_name) else os.path.join("results", run_name)
            if not os.path.exists(run_path):
                print(f"Warning: Run {run_path} not found, skipping.")
                continue
            all_results = load_run_results(run_path)
            df = build_dataframe(all_results)
            all_dfs.append(df)
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            generate_csv_summary(combined_df, args.output_dir)
            if not args.no_plots:
                plot_accuracy_vs_rank(combined_df, args.output_dir)
                plot_model_comparison(combined_df, args.output_dir)
                # For each run, also plot training curves if logs exist
                for run_name in args.compare_runs:
                    run_path = run_name if os.path.exists(run_name) else os.path.join("results", run_name)
                    plot_training_curves(run_path, args.output_dir)
        else:
            print("No valid runs to compare.")
    
    else:
        print("Please provide either --results_dir or --compare_runs")
        sys.exit(1)
    
    print(f"\n✅ Analysis complete. Outputs saved in: {args.output_dir}")
    print("  - CSV files: all_results.csv, per‑task CSV")
    print("  - Plots: rank vs metric, model comparison, training curves")


if __name__ == "__main__":
    main()
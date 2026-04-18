#!/usr/bin/env python3
"""
GLUE Benchmark Evaluation Script for SOA Models

This script evaluates multiple SOA model checkpoints (with different ranks) on
all selected GLUE tasks. It saves detailed results, generates a comprehensive
summary, and creates a suite of professional visualizations.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

# Hugging Face libraries
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

# Add the project root to the path so we can import SOA modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from soa.config import SOAConfig
from soa.model import SOABert
from soa.utils import get_device, set_seed

# =============================================================================
# Configuration
# =============================================================================

# All available GLUE tasks
GLUE_TASKS = [
    "cola",          # Single sentence acceptability (MCC)
    "mnli",          # NLI (accuracy, matched/mismatched)
    "mrpc",          # Paraphrase detection (accuracy, f1)
    "qnli",          # QA/NLI (accuracy)
    "qqp",           # Quora question pairs (accuracy, f1)
    "rte",           # Textual entailment (accuracy)
    "sst2",          # Sentiment analysis (accuracy)
    "stsb",          # Semantic textual similarity (Pearson/Spearman)
    "wnli",          # Winograd NLI (accuracy)
]

# Tasks with two evaluation sets
MNLI_TASKS = ["mnli_matched", "mnli_mismatched"]

# Task-specific metrics
TASK_METRICS = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr"],
    "wnli": ["accuracy"],
}

# Maximum sequence length for tokenization
MAX_SEQ_LENGTH = 128

# Batch size for evaluation (adjust based on your M1 memory)
EVAL_BATCH_SIZE = 32


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def load_glue_dataset(task_name: str, split: str = "validation"):
    """
    Load a GLUE dataset using the Hugging Face datasets library.
    
    Args:
        task_name: GLUE task name (e.g., 'cola', 'mrpc')
        split: 'train', 'validation', or 'test'
    
    Returns:
        Dataset object
    """
    print(f"    Loading {task_name} ({split})...")
    try:
        # Some tasks have specific configs
        if task_name == "mnli":
            # For MNLI, we need to handle matched/mismatched separately
            dataset = load_dataset("glue", "mnli", split=f"validation_{split}")
        else:
            dataset = load_dataset("glue", task_name, split=split)
        return dataset
    except Exception as e:
        print(f"    Error loading {task_name}: {e}")
        # Try with 'validation' split for MNLI
        if task_name == "mnli":
            try:
                dataset = load_dataset("glue", "mnli", split="validation_matched")
                return dataset
            except:
                pass
        raise


def prepare_dataloader(dataset, tokenizer, task_name: str, batch_size: int = 32):
    """
    Convert a GLUE dataset into a PyTorch DataLoader.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Determine the text fields based on the task
    if task_name in ["cola", "sst2"]:
        text_key = "sentence"
    elif task_name in ["mrpc", "qqp"]:
        text_key_1, text_key_2 = "sentence1", "sentence2"
    elif task_name in ["mnli", "qnli", "rte", "wnli"]:
        text_key_1, text_key_2 = "premise", "hypothesis"
    elif task_name == "stsb":
        text_key_1, text_key_2 = "sentence1", "sentence2"
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for example in dataset:
        if task_name in ["cola", "sst2"]:
            texts.append(example[text_key])
        else:
            texts.append((example[text_key_1], example[text_key_2]))
        
        # Handle labels
        if task_name == "stsb":
            # STS-B has continuous labels (0-5)
            labels.append(float(example["label"]))
        elif task_name == "cola":
            # CoLA has labels 0 (unacceptable) and 1 (acceptable)
            labels.append(example["label"])
        else:
            labels.append(example["label"])
    
    # Tokenize
    print(f"    Tokenizing {len(texts)} examples...")
    
    if task_name in ["cola", "sst2"]:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
    else:
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
    
    # Convert labels to tensor
    if task_name == "stsb":
        label_tensor = torch.tensor(labels, dtype=torch.float)
    else:
        label_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    tensor_dataset = TensorDataset(
        encodings["input_ids"],
        encodings.get("token_type_ids", torch.zeros_like(encodings["input_ids"])),
        encodings["attention_mask"],
        label_tensor
    )
    
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def evaluate_model_on_task(
    model: torch.nn.Module,
    dataloader: DataLoader,
    task_name: str,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a model on a single GLUE task.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"      Batches", leave=False):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3]
            
            logits = model(input_ids, token_type_ids, attention_mask)
            
            if task_name == "stsb":
                # STS-B: regression task
                preds = logits.squeeze(-1).cpu().numpy()
            else:
                # Classification tasks
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds.tolist() if isinstance(preds, np.ndarray) else preds)
            all_labels.extend(labels.numpy().tolist())
    
    # Calculate metrics
    results = {}
    
    if task_name == "cola":
        results["matthews_correlation"] = matthews_corrcoef(all_labels, all_preds)
    elif task_name == "stsb":
        from scipy.stats import pearsonr, spearmanr
        results["pearson"] = pearsonr(all_labels, all_preds)[0]
        results["spearmanr"] = spearmanr(all_labels, all_preds)[0]
    elif task_name in ["mrpc", "qqp"]:
        results["accuracy"] = accuracy_score(all_labels, all_preds)
        results["f1"] = f1_score(all_labels, all_preds, average="binary" if task_name == "mrpc" else "weighted")
    else:
        results["accuracy"] = accuracy_score(all_labels, all_preds)
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def setup_plotting_style():
    """Set consistent, professional plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def create_bar_chart(df: pd.DataFrame, metric: str, output_path: Path):
    """
    Create a grouped bar chart comparing models across tasks.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Pivot data for plotting
    pivot_df = df.pivot(index="task", columns="model", values=metric)
    
    # Sort tasks alphabetically
    pivot_df = pivot_df.sort_index()
    
    # Plot
    pivot_df.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_title(f"GLUE Benchmark: {metric.upper()} by Task and Model Rank", fontweight="bold")
    ax.set_xlabel("GLUE Task")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Model (Rank)", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved bar chart: {output_path}")


def create_heatmap(df: pd.DataFrame, metric: str, output_path: Path):
    """
    Create a heatmap showing model performance across tasks.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot data
    pivot_df = df.pivot(index="model", columns="task", values=metric)
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(f"GLUE Benchmark: {metric.upper()} Heatmap", fontweight="bold", fontsize=14)
    ax.set_xlabel("GLUE Task")
    ax.set_ylabel("Model (Rank)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved heatmap: {output_path}")


def create_radar_chart(df: pd.DataFrame, output_path: Path):
    """
    Create a radar chart comparing models across all tasks.
    """
    from math import pi
    
    # Prepare data
    pivot_df = df.pivot(index="task", columns="model", values="primary_metric")
    tasks = pivot_df.index.tolist()
    models = pivot_df.columns.tolist()
    
    # Number of variables
    N = len(tasks)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot each model
    colors = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        values = pivot_df[model].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add task labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    ax.set_title("GLUE Benchmark: Model Comparison Radar Chart", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved radar chart: {output_path}")


def create_rank_vs_accuracy_plot(df: pd.DataFrame, output_path: Path):
    """
    Create a line plot showing how accuracy varies with rank across tasks.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract rank from model name
    df_plot = df.copy()
    df_plot['rank'] = df_plot['model'].str.extract(r'rank_(\d+)').astype(int)
    
    # Sort by rank
    df_plot = df_plot.sort_values('rank')
    
    # Plot line for each task
    for task in df_plot['task'].unique():
        task_df = df_plot[df_plot['task'] == task]
        ax.plot(task_df['rank'], task_df['primary_metric'], 'o-', label=task, linewidth=2, markersize=6)
    
    ax.set_title("Effect of Rank on GLUE Task Performance", fontweight="bold")
    ax.set_xlabel("Rank (k_rank)")
    ax.set_ylabel("Primary Metric")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved rank vs accuracy plot: {output_path}")


def create_summary_table(df: pd.DataFrame, output_path: Path):
    """
    Create a nicely formatted summary table and save as CSV.
    """
    # Pivot for a clean summary
    summary = df.pivot_table(
        index='task',
        columns='model',
        values='primary_metric',
        aggfunc='first'
    )
    
    # Add average row
    summary.loc['AVERAGE'] = summary.mean()
    
    # Round to 4 decimal places
    summary = summary.round(4)
    
    # Save to CSV
    summary.to_csv(output_path)
    print(f"  ✅ Saved summary table: {output_path}")
    
    return summary


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate SOA models on GLUE benchmark")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory containing model .pt files")
    parser.add_argument("--model_pattern", type=str, default="soa_bert_rank_*.pt",
                        help="Glob pattern for model files")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="GLUE tasks to evaluate (default: all except mnli mismatched)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base directory for results")
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = get_device()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"glue_benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    details_dir = run_dir / "task_details"
    details_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 SOA GLUE Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Results will be saved to: {run_dir}")
    print(f"Device: {device}")
    print()
    
    # Find model files
    models_dir = Path(args.models_dir)
    model_files = sorted(models_dir.glob(args.model_pattern))
    
    if not model_files:
        print(f"❌ No model files found matching '{args.model_pattern}' in {models_dir}")
        return
    
    print(f"📦 Found {len(model_files)} model(s):")
    for mf in model_files:
        print(f"   - {mf.name}")
    print()
    
    # Determine tasks to evaluate
    if args.tasks:
        tasks = args.tasks
    else:
        # Default: all GLUE tasks except mnli_mismatched (to save time)
        tasks = [t for t in GLUE_TASKS if t != "mnli"] + ["mnli_matched"]
    
    print(f"📋 Tasks to evaluate: {tasks}")
    print()
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Store all results
    all_results = []
    
    # Evaluate each model
    for model_idx, model_path in enumerate(model_files):
        # Extract rank from filename
        import re
        rank_match = re.search(r'rank_(\d+)', model_path.name)
        rank = int(rank_match.group(1)) if rank_match else 64
        model_name = f"SOA_rank_{rank}"
        
        print(f"\n{'='*50}")
        print(f"🔵 Evaluating Model {model_idx+1}/{len(model_files)}: {model_name}")
        print(f"{'='*50}")
        
        # Load model
        cfg = SOAConfig(k_rank=rank)
        model = SOABert(cfg)
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"  ✅ Model loaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue
        
        model.to(device)
        model.eval()
        
        # Evaluate on each task
        for task in tasks:
            print(f"\n  📊 Evaluating on {task.upper()}...")
            
            try:
                # Handle MNLI matched/mismatched
                if task == "mnli_matched":
                    dataset = load_glue_dataset("mnli", "matched")
                elif task == "mnli_mismatched":
                    dataset = load_glue_dataset("mnli", "mismatched")
                else:
                    dataset = load_glue_dataset(task, "validation")
                
                # Prepare dataloader
                dataloader = prepare_dataloader(dataset, tokenizer, task.split('_')[0], args.batch_size)
                
                # Evaluate
                start_time = time.time()
                task_results = evaluate_model_on_task(model, dataloader, task.split('_')[0], device)
                eval_time = time.time() - start_time
                
                # Determine primary metric
                if task.startswith("cola"):
                    primary_metric = task_results.get("matthews_correlation", 0)
                    primary_name = "matthews_correlation"
                elif task.startswith("stsb"):
                    primary_metric = task_results.get("pearson", 0)
                    primary_name = "pearson"
                else:
                    primary_metric = task_results.get("accuracy", 0)
                    primary_name = "accuracy"
                
                # Store results
                result_entry = {
                    "model": model_name,
                    "rank": rank,
                    "task": task,
                    "primary_metric": primary_metric,
                    "primary_metric_name": primary_name,
                    "all_metrics": task_results,
                    "eval_time_seconds": eval_time,
                    "num_examples": len(dataset)
                }
                all_results.append(result_entry)
                
                # Save detailed task results
                task_detail_path = details_dir / f"{model_name}_{task}.json"
                with open(task_detail_path, 'w') as f:
                    json.dump(result_entry, f, indent=2, default=str)
                
                print(f"      {primary_name}: {primary_metric:.4f} (time: {eval_time:.2f}s)")
                
            except Exception as e:
                print(f"      ❌ Error evaluating {task}: {e}")
                continue
    
    # Save raw results
    raw_results_path = run_dir / "raw_results.json"
    with open(raw_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Raw results saved to: {raw_results_path}")
    
    # Convert to DataFrame for analysis
    if not all_results:
        print("❌ No results collected. Exiting.")
        return
    
    df = pd.DataFrame(all_results)
    
    # Create summary CSV
    summary_df = df[["model", "task", "primary_metric"]].copy()
    summary_csv_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Summary CSV saved to: {summary_csv_path}")
    
    # Create formatted summary table
    summary_table = create_summary_table(df, run_dir / "summary_table.csv")
    print("\n📊 Summary Table:")
    print(summary_table)
    
    # Generate plots
    if not args.skip_plots:
        print("\n🎨 Generating visualizations...")
        setup_plotting_style()
        
        # 1. Bar chart for primary metrics
        create_bar_chart(df, "primary_metric", plots_dir / "primary_metric_bar_chart.png")
        
        # 2. Heatmap
        create_heatmap(df, "primary_metric", plots_dir / "primary_metric_heatmap.png")
        
        # 3. Radar chart
        create_radar_chart(df, plots_dir / "radar_chart.png")
        
        # 4. Rank vs accuracy plot
        create_rank_vs_accuracy_plot(df, plots_dir / "rank_vs_accuracy.png")
        
        print("\n✅ All visualizations saved to:", plots_dir)
    
    print(f"\n{'='*60}")
    print(f"🎉 GLUE Benchmark Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results directory: {run_dir}")
    print(f"  - Raw results: raw_results.json")
    print(f"  - Summary CSV: summary.csv")
    print(f"  - Summary table: summary_table.csv")
    if not args.skip_plots:
        print(f"  - Plots: {plots_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
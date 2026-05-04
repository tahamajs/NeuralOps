# framework/analysis/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

def set_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)

def create_comprehensive_plots(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (18, 16)
) -> None:
    """
    Create the full 3x3 grid of analysis plots.
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('SVD Compression Analysis: Comprehensive Results', fontsize=16, fontweight='bold')

    # 1. Accuracy vs Rank (by model)
    ax = axes[0,0]
    for model in df['model'].unique():
        sub = df[df['model'] == model].groupby('rank')['accuracy'].mean().reset_index()
        ax.plot(sub['rank'], sub['accuracy'], marker='o', label=model, linewidth=2)
    ax.set_title('Accuracy vs Rank (by Model)'); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Accuracy vs Rank (by task)
    ax = axes[0,1]
    for task in df['task'].unique():
        sub = df[df['task'] == task].groupby('rank')['accuracy'].mean().reset_index()
        ax.plot(sub['rank'], sub['accuracy'], marker='s', label=task, linewidth=2)
    ax.set_title('Accuracy vs Rank (by Task)'); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Speedup factor
    ax = axes[0,2]
    if 'speedup_factor' in df.columns:
        for model in df['model'].unique():
            sub = df[df['model'] == model].groupby('rank')['speedup_factor'].mean()
            ax.plot(sub.index, sub.values, marker='^', label=model, linewidth=2)
        ax.set_title('Speedup vs Rank'); ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No speedup data', ha='center', va='center', transform=ax.transAxes)

    # 4. Parameter reduction
    ax = axes[1,0]
    if 'param_reduction' in df.columns:
        pd_data = df.groupby(['model','rank'])['param_reduction'].mean().reset_index()
        models = pd_data['model'].unique()
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            sub = pd_data[pd_data['model'] == m]
            x = sub['rank'] + (i - len(models)/2) * width
            ax.bar(x, sub['param_reduction'], width=width, label=m)
        ax.set_title('Parameter Reduction'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5,0.5,'No param reduction', ha='center', va='center', transform=ax.transAxes)

    # 5. Accuracy drop distribution
    ax = axes[1,1]
    if 'accuracy_drop' in df.columns:
        drops = df['accuracy_drop'].dropna()
        ax.hist(drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax.axvline(drops.mean(), color='blue', linestyle='-', label=f'Mean: {drops.mean():.3f}')
        ax.set_title('Accuracy Drop Distribution'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5,0.5,'No accuracy drop', ha='center', va='center', transform=ax.transAxes)

    # 6. Memory reduction
    ax = axes[1,2]
    if 'memory_reduction' in df.columns:
        mem = df.groupby('rank')['memory_reduction'].mean().reset_index()
        ax.bar(mem['rank'], mem['memory_reduction'], color='lightgreen', edgecolor='darkgreen')
        ax.set_title('Avg Memory Reduction'); ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5,0.5,'No memory reduction', ha='center', va='center', transform=ax.transAxes)

    # 7. Accuracy by task type
    ax = axes[2,0]
    if 'task_type' in df.columns:
        for tt in df['task_type'].unique():
            sub = df[df['task_type'] == tt].groupby('rank')['accuracy'].mean().reset_index()
            ax.plot(sub['rank'], sub['accuracy'], marker='D', label=tt, linewidth=2)
        ax.set_title('Accuracy by Task Type'); ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5,0.5,'No task type', ha='center', va='center', transform=ax.transAxes)

    # 8. Compression ratio heatmap
    ax = axes[2,1]
    if 'compression_ratio' in df.columns:
        pivot = df.pivot_table(index='model', columns='rank', values='compression_ratio', aggfunc='mean')
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label='Compression Ratio')
        ax.set_title('Compression Ratio Heatmap')
    else:
        ax.text(0.5,0.5,'No comp ratio', ha='center', va='center', transform=ax.transAxes)

    # 9. Best config per dataset
    ax = axes[2,2]
    best = []
    for task in df['task'].unique():
        sub = df[df['task'] == task]
        if 'accuracy' in sub.columns:
            idx = sub['accuracy'].idxmax()
            best.append(sub.loc[idx])
    if best:
        best_df = pd.DataFrame(best)
        ax.barh(best_df['task'], best_df['accuracy'], color='skyblue', edgecolor='navy')
        ax.set_title('Best Configuration per Dataset')
        for i, (acc, model, rank) in enumerate(zip(best_df['accuracy'], best_df['model'], best_df['rank'])):
            ax.text(acc+0.005, i, f'{model} r={int(rank)}', va='center', fontsize=8)
    else:
        ax.text(0.5,0.5,'No best config', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive plots saved to {output_dir}")

def plot_tradeoff(df: pd.DataFrame, output_dir: str, metric: str = 'accuracy') -> None:
    """Scatter plot of Speedup vs Accuracy (or other metric)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in df['model'].unique():
        sub = df[df['model'] == model].dropna(subset=['speedup_factor', metric])
        ax.scatter(sub['speedup_factor'], sub[metric], s=100, label=model, alpha=0.7)
        for _, row in sub.iterrows():
            ax.annotate(f"r={int(row['rank'])}", (row['speedup_factor'], row[metric]), fontsize=8)
    ax.set_xlabel('Speedup Factor'); ax.set_ylabel(metric.capitalize())
    ax.set_title('Accuracy vs Speedup Tradeoff')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(y=df[metric].max(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_pareto_front(df: pd.DataFrame, output_dir: str, 
                      x_metric: str = 'speedup_factor', y_metric: str = 'accuracy') -> None:
    """Highlight Pareto‑optimal points (non‑dominated)."""
    set_style()
    points = df[[x_metric, y_metric, 'model', 'rank']].dropna().values
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    pareto_df = pd.DataFrame(pareto, columns=[x_metric, y_metric, 'model', 'rank'])
    fig, ax = plt.subplots(figsize=(10,8))
    for model in df['model'].unique():
        sub = df[df['model'] == model]
        ax.scatter(sub[x_metric], sub[y_metric], label=model, alpha=0.5)
    ax.scatter(pareto_df[x_metric], pareto_df[y_metric], color='red', s=150, marker='*', label='Pareto front')
    for _, row in pareto_df.iterrows():
        ax.annotate(f"{row['model']} r={int(row['rank'])}", (row[x_metric], row[y_metric]), fontsize=8)
    ax.set_xlabel(x_metric.replace('_',' ').capitalize())
    ax.set_ylabel(y_metric.capitalize())
    ax.set_title('Pareto Frontier of Configurations')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pareto_front.png', dpi=150)
    plt.close()

def plot_rank_allocation(recommended_ranks: List[int], output_dir: str) -> None:
    """Bar plot of recommended rank per layer."""
    set_style()
    fig, ax = plt.subplots(figsize=(12,6))
    layers = list(range(len(recommended_ranks)))
    ax.bar(layers, recommended_ranks, color='steelblue', edgecolor='navy')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Recommended Rank')
    ax.set_title('Layer‑wise Rank Allocation Recommendation')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rank_allocation.png', dpi=150)
    plt.close()

def plot_training_history(history: dict, save_path: str) -> None:
    """Plot training and evaluation loss curves."""
    fig, ax = plt.subplots(figsize=(8,5))
    if history.get("train_steps") and history.get("train_loss"):
        ax.plot(history["train_steps"], history["train_loss"], label="Train Loss", color="blue")
    if history.get("eval_steps") and history.get("eval_loss"):
        ax.plot(history["eval_steps"], history["eval_loss"], label="Eval Loss", color="red", marker='o')
    ax.set_xlabel("Steps"); ax.set_ylabel("Loss")
    ax.set_title("Training History"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
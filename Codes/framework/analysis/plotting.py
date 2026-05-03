"""
Comprehensive plotting for experiment analysis.
All plots are saved in the specified directory (auto‑created).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import os

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_comprehensive_plots(results_df: pd.DataFrame, plots_dir: str, 
                                figsize: tuple = (18, 16)) -> None:
    """Create the full 3x3 grid of analysis plots."""
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('SVD Compression Analysis: Comprehensive Results', fontsize=16, fontweight='bold')

    # 1. Accuracy vs Rank (by model)
    ax1 = axes[0, 0]
    for model in results_df['model'].unique():
        md = results_df[results_df['model'] == model]
        if 'accuracy' in md.columns:
            md_grp = md.groupby('rank')['accuracy'].mean().reset_index()
            ax1.plot(md_grp['rank'], md_grp['accuracy'], marker='o', label=model, linewidth=2)
    ax1.set_title('Accuracy vs Rank (by Model)'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Accuracy vs Rank (by task)
    ax2 = axes[0, 1]
    for task in results_df['task'].unique():
        td = results_df[results_df['task'] == task]
        if 'accuracy' in td.columns:
            td_grp = td.groupby('rank')['accuracy'].mean().reset_index()
            ax2.plot(td_grp['rank'], td_grp['accuracy'], marker='s', label=task, linewidth=2)
    ax2.set_title('Accuracy vs Rank (by Task)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    # 3. Speedup factor
    ax3 = axes[0, 2]
    if 'speedup_factor' in results_df.columns:
        for model in results_df['model'].unique():
            md = results_df[results_df['model'] == model]
            sp = md.groupby('rank')['speedup_factor'].mean()
            ax3.plot(sp.index, sp.values, marker='^', label=model, linewidth=2)
        ax3.set_title('Speedup vs Rank'); ax3.legend(); ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Speedup data not available', ha='center', va='center', transform=ax3.transAxes)

    # 4. Parameter reduction (bar)
    ax4 = axes[1, 0]
    if 'param_reduction' in results_df.columns:
        pd_data = results_df.groupby(['model', 'rank'])['param_reduction'].mean().reset_index()
        models = pd_data['model'].unique()
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            md = pd_data[pd_data['model'] == m]
            ax4.bar(md['rank'] + (i - len(models)/2)*width, md['param_reduction'], width=width, label=m)
        ax4.set_title('Parameter Reduction by Model'); ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Param reduction data not available', ha='center', va='center', transform=ax4.transAxes)

    # 5. Accuracy drop distribution
    ax5 = axes[1, 1]
    if 'accuracy_drop' in results_df.columns:
        drops = results_df['accuracy_drop'].dropna()
        ax5.hist(drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax5.set_title('Accuracy Drop Distribution'); ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Accuracy drop data not available', ha='center', va='center', transform=ax5.transAxes)

    # 6. Memory reduction
    ax6 = axes[1, 2]
    if 'memory_reduction' in results_df.columns:
        mem = results_df.groupby('rank')['memory_reduction'].mean().reset_index()
        ax6.bar(mem['rank'], mem['memory_reduction'], color='lightgreen', edgecolor='darkgreen')
        ax6.set_title('Average Memory Reduction'); ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'Memory reduction data not available', ha='center', va='center', transform=ax6.transAxes)

    # 7. Accuracy by task type
    ax7 = axes[2, 0]
    if 'task_type' in results_df.columns:
        for tt in results_df['task_type'].unique():
            ttd = results_df[results_df['task_type'] == tt]
            avg = ttd.groupby('rank')['accuracy'].mean().reset_index()
            ax7.plot(avg['rank'], avg['accuracy'], marker='D', label=tt, linewidth=2)
        ax7.set_title('Accuracy by Task Type'); ax7.legend(); ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Task type data not available', ha='center', va='center', transform=ax7.transAxes)

    # 8. Compression ratio heatmap
    ax8 = axes[2, 1]
    if 'compression_ratio' in results_df.columns:
        pivot = results_df.pivot_table(index='model', columns='rank', values='compression_ratio', aggfunc='mean')
        im = ax8.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax8.set_xticks(range(len(pivot.columns))); ax8.set_xticklabels(pivot.columns)
        ax8.set_yticks(range(len(pivot.index))); ax8.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax8, label='Compression Ratio')
        ax8.set_title('Compression Ratio Heatmap')
    else:
        ax8.text(0.5, 0.5, 'Compression ratio data not available', ha='center', va='center', transform=ax8.transAxes)

    # 9. Best config per dataset
    ax9 = axes[2, 2]
    best = []
    for ds in results_df['task'].unique():
        ds_df = results_df[results_df['task'] == ds]
        if 'accuracy' in ds_df.columns:
            idx = ds_df['accuracy'].idxmax()
            best.append(ds_df.loc[idx])
    if best:
        best_df = pd.DataFrame(best)
        ax9.barh(best_df['task'], best_df['accuracy'], color='skyblue', edgecolor='navy')
        ax9.set_title('Best Configuration per Dataset')
    else:
        ax9.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax9.transAxes)

    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comprehensive plot to {plots_dir}")


def plot_training_history(history: dict, save_path: str):
    """Plot training and evaluation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if history.get("train_steps") and history.get("train_loss"):
        ax.plot(history["train_steps"], history["train_loss"], label="Train Loss", color="blue")
    if history.get("eval_steps") and history.get("eval_loss"):
        ax.plot(history["eval_steps"], history["eval_loss"], label="Eval Loss", color="red", marker="o")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tradeoff(results_df: pd.DataFrame, plots_dir: str):
    """Accuracy vs speedup scatter plot."""
    if 'speedup_factor' not in results_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in results_df['model'].unique():
        md = results_df[results_df['model'] == model]
        if 'accuracy' in md and 'speedup_factor' in md:
            ax.scatter(md['speedup_factor'], md['accuracy'], s=100, label=model, alpha=0.7)
    ax.set_xlabel("Speedup Factor")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Speedup Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(Path(plots_dir) / "tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
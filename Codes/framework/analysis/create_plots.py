"""
Comprehensive Visualization Generator for SVD Compression Results
Creates publication-quality plots for paper/report
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
import os


def create_comprehensive_plots(results_df: pd.DataFrame, plots_dir: str, 
                                figsize: tuple = (18, 16)) -> None:
    """
    Create comprehensive set of plots for SVD compression analysis.
    
    Args:
        results_df: DataFrame with columns: model, rank, task, accuracy, 
                   (optional: speedup_factor, param_reduction, memory_reduction)
        plots_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create main figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('SVD Compression Analysis: Comprehensive Results', 
                 fontsize=16, fontweight='bold')
    
    # Ensure required columns exist
    required_cols = ['rank', 'model', 'task']
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            return
    
    # 1. Accuracy vs Rank (by Model)
    ax1 = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if 'accuracy' in model_data.columns:
            model_data = model_data.groupby('rank')['accuracy'].mean().reset_index()
            ax1.plot(model_data['rank'], model_data['accuracy'], 
                    marker='o', label=model, linewidth=2, markersize=8)
    ax1.set_xlabel('SVD Rank')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Rank (by Model)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy vs Rank (by Task)
    ax2 = axes[0, 1]
    for task in results_df['task'].unique():
        task_data = results_df[results_df['task'] == task]
        if 'accuracy' in task_data.columns:
            task_data = task_data.groupby('rank')['accuracy'].mean().reset_index()
            ax2.plot(task_data['rank'], task_data['accuracy'], 
                    marker='s', label=task, linewidth=2, markersize=8)
    ax2.set_xlabel('SVD Rank')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Rank (by Dataset)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Speedup factor (if available)
    ax3 = axes[0, 2]
    if 'speedup_factor' in results_df.columns:
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            speedup_by_rank = model_data.groupby('rank')['speedup_factor'].mean()
            ax3.plot(speedup_by_rank.index, speedup_by_rank.values, 
                    marker='^', label=model, linewidth=2, markersize=8)
        ax3.set_xlabel('SVD Rank')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Inference Speedup vs Rank')
        ax3.legend(loc='best', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Speedup data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter reduction (bar chart)
    ax4 = axes[1, 0]
    if 'param_reduction' in results_df.columns:
        param_data = results_df.groupby(['model', 'rank'])['param_reduction'].mean().reset_index()
        models = param_data['model'].unique()
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            model_data = param_data[param_data['model'] == model]
            x = model_data['rank'] + (i - len(models)/2) * width
            ax4.bar(x, model_data['param_reduction'], width=width, label=model)
        ax4.set_xlabel('SVD Rank')
        ax4.set_ylabel('Parameter Reduction')
        ax4.set_title('Parameter Reduction by Model')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Parameter reduction data not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Accuracy drop distribution
    ax5 = axes[1, 1]
    if 'accuracy_drop' in results_df.columns:
        accuracy_drops = results_df['accuracy_drop'].dropna()
        ax5.hist(accuracy_drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax5.axvline(x=accuracy_drops.mean(), color='blue', linestyle='-', 
                   linewidth=2, label=f'Mean: {accuracy_drops.mean():.3f}')
        ax5.set_xlabel('Accuracy Drop')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Accuracy Drops')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Accuracy drop data not available', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Memory reduction
    ax6 = axes[1, 2]
    if 'memory_reduction' in results_df.columns:
        memory_data = results_df.groupby('rank')['memory_reduction'].mean().reset_index()
        ax6.bar(memory_data['rank'], memory_data['memory_reduction'], 
               color='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax6.set_xlabel('SVD Rank')
        ax6.set_ylabel('Memory Reduction (MB)')
        ax6.set_title('Average Memory Reduction')
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'Memory reduction data not available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    # 7. Accuracy by task type (if task_type available)
    ax7 = axes[2, 0]
    if 'task_type' in results_df.columns:
        for task_type in results_df['task_type'].unique():
            task_type_data = results_df[results_df['task_type'] == task_type]
            if 'accuracy' in task_type_data.columns:
                avg_by_rank = task_type_data.groupby('rank')['accuracy'].mean().reset_index()
                ax7.plot(avg_by_rank['rank'], avg_by_rank['accuracy'], 
                        marker='D', label=task_type, linewidth=2, markersize=8)
        ax7.set_xlabel('SVD Rank')
        ax7.set_ylabel('Accuracy')
        ax7.set_title('Accuracy by Task Type')
        ax7.legend(loc='best', fontsize=8)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Task type data not available', 
                ha='center', va='center', transform=ax7.transAxes)
    
    # 8. Compression ratio heatmap
    ax8 = axes[2, 1]
    if 'compression_ratio' in results_df.columns:
        pivot_data = results_df.pivot_table(
            index='model', columns='rank', values='compression_ratio', aggfunc='mean'
        )
        im = ax8.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        ax8.set_xticks(range(len(pivot_data.columns)))
        ax8.set_xticklabels(pivot_data.columns)
        ax8.set_yticks(range(len(pivot_data.index)))
        ax8.set_yticklabels(pivot_data.index)
        ax8.set_xlabel('SVD Rank')
        ax8.set_ylabel('Model')
        ax8.set_title('Compression Ratio Heatmap')
        plt.colorbar(im, ax=ax8, label='Compression Ratio')
    else:
        ax8.text(0.5, 0.5, 'Compression ratio data not available', 
                ha='center', va='center', transform=ax8.transAxes)
    
    # 9. Best configuration per dataset
    ax9 = axes[2, 2]
    best_configs = []
    for dataset in results_df['task'].unique():
        dataset_data = results_df[results_df['task'] == dataset]
        if 'accuracy' in dataset_data.columns:
            best_idx = dataset_data['accuracy'].idxmax()
            best_configs.append(dataset_data.loc[best_idx])
    
    if best_configs:
        best_df = pd.DataFrame(best_configs)
        y_pos = np.arange(len(best_df))
        ax9.barh(y_pos, best_df['accuracy'], color='skyblue', edgecolor='navy')
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(best_df['task'])
        ax9.set_xlabel('Accuracy')
        ax9.set_title('Best Configuration for Each Dataset')
        ax9.invert_yaxis()
        
        # Add labels
        for i, (acc, model, rank) in enumerate(zip(best_df['accuracy'], 
                                                    best_df['model'], 
                                                    best_df['rank'])):
            ax9.text(acc + 0.01, i, f'{model} (R{rank})', va='center', fontsize=8)
    else:
        ax9.text(0.5, 0.5, 'No best config data available', 
                ha='center', va='center', transform=ax9.transAxes)
    
    plt.tight_layout()
    output_path = Path(plots_dir) / 'comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive plots saved to {output_path}")
    
    # Also create individual plots
    create_individual_plots(results_df, plots_dir)


def create_individual_plots(df: pd.DataFrame, plots_dir: str) -> None:
    """Create individual detailed plots for paper."""
    
    # 1. Accuracy comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(index='model', columns='rank', values='accuracy', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title('Model Accuracy Comparison by Rank')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Rank')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'accuracy_by_model.png', dpi=150)
    plt.close()
    
    # 2. Radar chart for model comparison (if multiple models and tasks)
    if len(df['model'].unique()) >= 2 and len(df['task'].unique()) >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        tasks = df['task'].unique()[:6]
        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            values = []
            for task in tasks:
                task_acc = model_data[model_data['task'] == task]['accuracy'].mean()
                values.append(task_acc if not np.isnan(task_acc) else 0)
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(Path(plots_dir) / 'radar_chart.png', dpi=150)
        plt.close()
    
    # 3. Rank efficiency plot (accuracy per parameter)
    if 'param_reduction' in df.columns and 'accuracy' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax.scatter(model_data['param_reduction'], model_data['accuracy'], 
                      label=model, s=100, alpha=0.7)
            
            # Add rank labels
            for _, row in model_data.iterrows():
                ax.annotate(f"r={int(row['rank'])}", 
                           (row['param_reduction'], row['accuracy']),
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Parameter Reduction (%)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Efficiency: Accuracy vs Parameter Reduction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(plots_dir) / 'efficiency_plot.png', dpi=150)
        plt.close()


def create_tradeoff_plot(results_df: pd.DataFrame, plots_dir: str) -> None:
    """
    Create accuracy vs speedup tradeoff plot.
    """
    if 'speedup_factor' not in results_df.columns:
        print("Speedup data not available for tradeoff plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if 'accuracy' in model_data.columns and 'speedup_factor' in model_data.columns:
            # Get full attention baseline (rank=0 or highest accuracy)
            full_acc = model_data[model_data['rank'] == model_data['rank'].max()]['accuracy'].mean()
            if np.isnan(full_acc):
                full_acc = model_data['accuracy'].max()
            
            # Plot each rank
            for _, row in model_data.iterrows():
                if np.isnan(row['accuracy']) or np.isnan(row['speedup_factor']):
                    continue
                acc_drop = full_acc - row['accuracy']
                ax.scatter(row['speedup_factor'], acc_drop, 
                          s=100, label=f"{model} r={int(row['rank'])}" if _ == 0 else "")
                ax.annotate(f"r={int(row['rank'])}", 
                           (row['speedup_factor'], acc_drop),
                           fontsize=8)
    
    ax.set_xlabel('Speedup Factor')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('Accuracy-Speedup Tradeoff')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'tradeoff_plot.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comprehensive plots from results")
    parser.add_argument("--results_csv", type=str, required=True, 
                       help="Path to combined results CSV")
    parser.add_argument("--plots_dir", type=str, default="./plots", 
                       help="Directory to save plots")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")
    print(f"Columns: {list(df.columns)}")
    
    # Create plots
    create_comprehensive_plots(df, args.plots_dir)
    create_tradeoff_plot(df, args.plots_dir)
    
    print(f"\n✅ All plots saved to {args.plots_dir}")
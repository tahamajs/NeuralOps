# framework/analysis/visualization.py
"""
Comprehensive visualization functions for SVD compression results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import os


def create_comprehensive_plots(
    results_df: pd.DataFrame, 
    plots_dir: str, 
    figsize: Tuple[int, int] = (18, 16)
) -> None:
    """
    Create comprehensive set of plots for SVD compression analysis.
    
    Args:
        results_df: DataFrame with columns: model, rank, task, accuracy
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
    
    # 3-9 continue similarly...
    
    plt.tight_layout()
    output_path = Path(plots_dir) / 'comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive plots saved to {output_path}")


def create_individual_plots(df: pd.DataFrame, plots_dir: str) -> None:
    """Create individual detailed plots for paper."""
    
    # Accuracy comparison bar plot
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
    
    # Radar chart for model comparison
    if len(df['model'].unique()) >= 2 and len(df['task'].unique()) >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        tasks = df['task'].unique()[:6]
        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        angles += angles[:1]
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            values = []
            for task in tasks:
                task_acc = model_data[model_data['task'] == task]['accuracy'].mean()
                values.append(task_acc if not np.isnan(task_acc) else 0)
            values += values[:1]
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


def create_tradeoff_plot(results_df: pd.DataFrame, plots_dir: str) -> None:
    """Create accuracy vs speedup tradeoff plot."""
    if 'speedup_factor' not in results_df.columns:
        print("Speedup data not available for tradeoff plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if 'accuracy' in model_data.columns and 'speedup_factor' in model_data.columns:
            full_acc = model_data[model_data['rank'] == model_data['rank'].max()]['accuracy'].mean()
            if np.isnan(full_acc):
                full_acc = model_data['accuracy'].max()
            
            for _, row in model_data.iterrows():
                if np.isnan(row['accuracy']) or np.isnan(row['speedup_factor']):
                    continue
                acc_drop = full_acc - row['accuracy']
                ax.scatter(row['speedup_factor'], acc_drop, s=100)
                ax.annotate(f"r={int(row['rank'])}", 
                           (row['speedup_factor'], acc_drop), fontsize=8)
    
    ax.set_xlabel('Speedup Factor')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('Accuracy-Speedup Tradeoff')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'tradeoff_plot.png', dpi=150)
    plt.close()
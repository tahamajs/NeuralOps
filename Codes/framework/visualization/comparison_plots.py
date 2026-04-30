# framework/visualization/comparison_plots.py
"""
Comparison plotting utilities for model and rank analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from .utils import set_plot_style, save_figure


def plot_accuracy_vs_rank(
    df: pd.DataFrame,
    task: Optional[str] = None,
    hue: str = 'model',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot accuracy vs compression rank.
    
    Args:
        df: DataFrame with 'rank', 'accuracy', and hue column
        task: Specific task to filter (optional)
        hue: Column to use for different lines
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    if task:
        df = df[df['task'] == task]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for group in df[hue].unique():
        group_data = df[df[hue] == group].sort_values('rank')
        ax.plot(group_data['rank'], group_data['accuracy'], 
                marker='o', label=group, linewidth=2, markersize=8)
    
    ax.set_xlabel('SVD Rank')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs Compression Rank' + (f' - {task.upper()}' if task else ''))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_radar_chart(
    df: pd.DataFrame,
    models: List[str],
    tasks: List[str],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create radar chart for model comparison across tasks.
    
    Args:
        df: DataFrame with results
        models: List of models to compare
        tasks: List of tasks to include
        metric: Metric to plot
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, colors):
        values = []
        for task in tasks:
            val = df[(df['model'] == model) & (df['task'] == task)][metric].mean()
            values.append(val if not np.isnan(val) else 0)
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
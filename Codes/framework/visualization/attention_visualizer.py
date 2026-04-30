# framework/visualization/attention_visualizer.py
"""
Attention visualization utilities for SVD compression analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import subspace_angles
from typing import Optional, List, Dict, Any
from .base_visualizer import BaseVisualizer
from ..core.registry import VISUALIZER_REGISTRY


class SVDAttentionVisualizer(BaseVisualizer):
    """
    Visualizer for attention mechanisms in SVD-compressed models.
    
    Provides methods for:
    - Attention heatmaps
    - Attention entropy plots
    - Difference maps between full and compressed attention
    - Singular value spectrum analysis
    - Subspace alignment
    - Layer attention grids
    - Rank and model comparisons
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        super().__init__(output_dir)
        self.colormap = LinearSegmentedColormap.from_list(
            'custom', ['#440154', '#3B528B', '#21908C', '#5DC963', '#FDE725']
        )
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for consistent plots."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        rank: int = 64,
        model_name: str = "model",
        task_name: str = "task",
        save: bool = True,
        show: bool = False
    ) -> Optional[str]:
        """
        Plot attention heatmap for a specific layer and head.
        
        Args:
            attention_matrix: Attention weights of shape (seq_len, seq_len)
            tokens: Token labels for axes
            layer_idx: Layer index
            head_idx: Head index
            rank: Compression rank
            model_name: Model name
            task_name: Task name
            save: Whether to save the figure
            show: Whether to display the figure
            
        Returns:
            Path to saved figure if save=True
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=self.colormap,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title(f'{model_name} - {task_name} (Rank {rank})\nLayer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        plt.tight_layout()
        
        filename = f"attention_heatmap_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        
        if save:
            return self._save_figure(fig, filename, dpi=150)
        
        if show:
            plt.show()
        
        return filename
    
    def plot_attention_entropy(
        self,
        attention_matrix: np.ndarray,
        layer_idx: int = 0,
        head_idx: int = 0,
        rank: int = 64,
        model_name: str = "model",
        task_name: str = "task",
        save: bool = True
    ) -> Optional[str]:
        """
        Plot attention entropy per token.
        
        Args:
            attention_matrix: Attention weights
            layer_idx: Layer index
            head_idx: Head index
            rank: Compression rank
            model_name: Model name
            task_name: Task name
            save: Whether to save
            
        Returns:
            Path to saved figure
        """
        epsilon = 1e-10
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + epsilon), axis=-1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(entropy)), entropy, alpha=0.7, color='#21908C')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attention Entropy')
        ax.set_title(f'{model_name} - {task_name} (Rank {rank})\nLayer {layer_idx}, Head {head_idx} - Attention Entropy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"entropy_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        
        if save:
            return self._save_figure(fig, filename, dpi=150)
        
        return filename
    
    def plot_attention_difference(
        self,
        full_attention: np.ndarray,
        svd_attention: np.ndarray,
        layer_idx: int = 0,
        head_idx: int = 0,
        rank: int = 64,
        model_name: str = "model",
        task_name: str = "task",
        save: bool = True
    ) -> Optional[str]:
        """
        Plot difference between full and compressed attention.
        
        Args:
            full_attention: Full attention weights
            svd_attention: SVD-compressed attention weights
            layer_idx: Layer index
            head_idx: Head index
            rank: Compression rank
            model_name: Model name
            task_name: Task name
            save: Whether to save
            
        Returns:
            Path to saved figure
        """
        difference = np.abs(full_attention - svd_attention)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Full attention
        sns.heatmap(full_attention, cmap=self.colormap, ax=axes[0], cbar=False)
        axes[0].set_title('Full Attention')
        
        # Compressed attention
        sns.heatmap(svd_attention, cmap=self.colormap, ax=axes[1], cbar=False)
        axes[1].set_title(f'SVD Attention (Rank {rank})')
        
        # Difference map
        sns.heatmap(difference, cmap='Reds', ax=axes[2], cbar_kws={'label': 'Absolute Difference'})
        axes[2].set_title('Difference Map')
        
        plt.suptitle(f'{model_name} - {task_name}\nLayer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        
        filename = f"attention_diff_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        
        if save:
            return self._save_figure(fig, filename, dpi=150)
        
        return filename
    
    def analyze_singular_values(
        self,
        attention_matrix: np.ndarray,
        rank: int,
        model_name: str = "model",
        task_name: str = "task",
        layer_idx: int = 0,
        head_idx: int = 0,
        save: bool = True
    ) -> tuple:
        """
        Analyze and plot singular value spectrum.
        
        Args:
            attention_matrix: Attention weights
            rank: Compression rank
            model_name: Model name
            task_name: Task name
            layer_idx: Layer index
            head_idx: Head index
            save: Whether to save
            
        Returns:
            Tuple of (singular_values, cumulative_energy)
        """
        U, s, Vt = np.linalg.svd(attention_matrix, full_matrices=False)
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Singular value spectrum
        axes[0].plot(s, 'o-', color='#3B528B', markersize=4, linewidth=1.5)
        axes[0].axvline(x=rank, color='red', linestyle='--', label=f'Rank {rank}')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Singular Value')
        axes[0].set_title('Singular Value Spectrum')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative energy
        axes[1].plot(cumulative_energy, 'o-', color='#21908C', markersize=4, linewidth=1.5)
        axes[1].axhline(y=0.95, color='green', linestyle='--', label='95% Energy')
        axes[1].axvline(x=rank, color='red', linestyle='--', label=f'Rank {rank}')
        axes[1].set_xlabel('Number of Singular Values')
        axes[1].set_ylabel('Cumulative Energy')
        axes[1].set_title('Cumulative Energy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - {task_name}\nLayer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        
        filename = f"svd_spectrum_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        
        if save:
            self._save_figure(fig, filename, dpi=150)
        
        return s, cumulative_energy
    
    # Additional methods: compute_subspace_alignment, plot_subspace_alignment,
    # plot_layer_analysis_grid, plot_rank_comparison, plot_model_comparison,
    # generate_training_plots continue in the same pattern...
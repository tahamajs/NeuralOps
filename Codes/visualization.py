import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch
from scipy.linalg import subspace_angles
import json
from datetime import datetime

class SVDAttentionVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colormap = LinearSegmentedColormap.from_list('custom', ['#440154', '#3B528B', '#21908C', '#5DC963', '#FDE725'])
        
    def extract_attention_weights(self, model, inputs, task_name, rank, model_name, layer_idx=0, head_idx=0):
        """Extract attention weights from the model."""
        # Set model to eval mode
        model.eval()
        
        # Register hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attention_weights.append(output[1].detach().cpu())
            elif hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights.extend([att.detach().cpu() for att in output.attentions])
        
        # Register hooks for all attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def plot_attention_heatmap(self, attention_matrix, tokens, layer_idx, head_idx, rank, model_name, task_name):
        """Generate a heatmap of attention weights."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(attention_matrix, 
                    xticklabels=tokens, 
                    yticklabels=tokens,
                    cmap=self.colormap,
                    cbar_kws={'label': 'Attention Weight'},
                    ax=ax)
        
        ax.set_title(f'{model_name} - {task_name} (Rank {rank})\nLayer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/attention_heatmap_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def plot_attention_entropy(self, attention_matrix, layer_idx, head_idx, rank, model_name, task_name):
        """Plot entropy of attention distribution (higher entropy = more diffuse attention)."""
        # Compute entropy per token
        epsilon = 1e-10
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + epsilon), axis=-1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(entropy)), entropy, alpha=0.7, color='#21908C')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attention Entropy')
        ax.set_title(f'{model_name} - {task_name} (Rank {rank})\nLayer {layer_idx}, Head {head_idx} - Attention Entropy')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/entropy_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def plot_attention_difference(self, full_attention, svd_attention, layer_idx, head_idx, rank, model_name, task_name):
        """Compare SVD-compressed attention to full attention."""
        difference = np.abs(full_attention - svd_attention)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Full attention
        sns.heatmap(full_attention, cmap=self.colormap, ax=axes[0], cbar=False)
        axes[0].set_title('Full Attention')
        
        # SVD attention
        sns.heatmap(svd_attention, cmap=self.colormap, ax=axes[1], cbar=False)
        axes[1].set_title(f'SVD Attention (Rank {rank})')
        
        # Difference
        sns.heatmap(difference, cmap='Reds', ax=axes[2], cbar_kws={'label': 'Absolute Difference'})
        axes[2].set_title('Difference Map')
        
        plt.suptitle(f'{model_name} - {task_name}\nLayer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        filename = f"{self.output_dir}/attention_diff_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def analyze_singular_values(self, attention_matrix, rank, model_name, task_name, layer_idx, head_idx):
        """Perform SVD analysis on the attention matrix."""
        U, s, Vt = np.linalg.svd(attention_matrix, full_matrices=False)
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Singular values
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
        filename = f"{self.output_dir}/svd_spectrum_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        
        return s, cumulative_energy
    
    def compute_subspace_alignment(self, full_attention, svd_attention):
        """Compute principal angles between subspaces of full and compressed attention."""
        U_full, _, _ = np.linalg.svd(full_attention, full_matrices=False)
        U_svd, _, _ = np.linalg.svd(svd_attention, full_matrices=False)
        
        min_dim = min(U_full.shape[1], U_svd.shape[1])
        U_full = U_full[:, :min_dim]
        U_svd = U_svd[:, :min_dim]
        
        angles = subspace_angles(U_full, U_svd)
        return np.degrees(angles)  # Convert to degrees
    
    def plot_subspace_alignment(self, angles, rank, model_name, task_name, layer_idx, head_idx):
        """Visualize principal angles between subspaces."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(angles, 'o-', color='#5DC963', markersize=6, linewidth=1.5)
        ax.set_xlabel('Principal Angle Index')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{model_name} - {task_name} (Rank {rank})\nLayer {layer_idx}, Head {head_idx} - Subspace Alignment')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/subspace_alignment_{model_name}_{task_name}_rank{rank}_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def plot_layer_analysis_grid(self, all_attention, rank, model_name, task_name, layers=[0, 4, 8, 11], heads=[0, 6, 11]):
        """Create a grid of attention heatmaps for multiple layers and heads."""
        fig, axes = plt.subplots(len(layers), len(heads), figsize=(16, 16))
        
        for i, layer in enumerate(layers):
            for j, head in enumerate(heads):
                if layer < len(all_attention) and head < all_attention[layer].shape[0]:
                    attn = all_attention[layer][head].numpy() if torch.is_tensor(all_attention[layer][head]) else all_attention[layer][head]
                    sns.heatmap(attn, cmap=self.colormap, ax=axes[i, j], cbar=False)
                    axes[i, j].set_title(f'Layer {layer}, Head {head}')
                    axes[i, j].set_xticklabels([])
                    axes[i, j].set_yticklabels([])
        
        plt.suptitle(f'{model_name} - {task_name} (Rank {rank})\nMulti-Layer Multi-Head Attention Grid')
        plt.tight_layout()
        filename = f"{self.output_dir}/attention_grid_{model_name}_{task_name}_rank{rank}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def plot_rank_comparison(self, results_dict, task_name):
        """Compare performance across different ranks."""
        ranks = sorted([r for r in results_dict.keys() if r != 'None' and 'accuracy' in results_dict[r]])
        accuracies = [results_dict[r]['accuracy'] for r in ranks if 'accuracy' in results_dict[r]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ranks, accuracies, 'o-', color='#440154', markersize=8, linewidth=2)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task_name} - Performance vs Compression Rank')
        ax.grid(True, alpha=0.3)
        
        # Add full attention baseline
        if 'None' in results_dict and 'accuracy' in results_dict['None']:
            ax.axhline(y=results_dict['None']['accuracy'], color='red', linestyle='--', label='Full Attention')
            ax.legend()
        
        plt.tight_layout()
        filename = f"{self.output_dir}/rank_comparison_{task_name}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def plot_model_comparison(self, all_model_results, task_name):
        """Compare multiple models on the same task."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#440154', '#3B528B', '#21908C', '#5DC963', '#FDE725']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (model_name, results) in enumerate(all_model_results.items()):
            ranks = [r for r in results.keys() if r != 'None']
            accuracies = [results[r]['accuracy'] for r in ranks]
            ax.plot(ranks, accuracies, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                   linewidth=2, markersize=8, label=model_name)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task_name} - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/model_comparison_{task_name}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    
    def generate_training_plots(self, training_logs, model_name, task_name, rank):
        """Generate training curves from logged metrics."""
        epochs = [log['epoch'] for log in training_logs]
        losses = [log['loss'] for log in training_logs]
        eval_scores = [log['eval_score'] for log in training_logs if 'eval_score' in log]
        eval_epochs = [log['epoch'] for log in training_logs if 'eval_score' in log]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        axes[0].plot(epochs, losses, 'o-', color='#3B528B', markersize=4, linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True, alpha=0.3)
        
        # Evaluation scores
        if eval_scores:
            axes[1].plot(eval_epochs, eval_scores, 'o-', color='#21908C', markersize=6, linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Evaluation Score')
            axes[1].set_title('Validation Performance')
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - {task_name} (Rank {rank})')
        plt.tight_layout()
        filename = f"{self.output_dir}/training_curves_{model_name}_{task_name}_rank{rank}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
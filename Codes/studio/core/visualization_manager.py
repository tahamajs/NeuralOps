# studio/core/visualization_manager.py
"""
Complete visualization system for model analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Tuple
import io
import base64
from PIL import Image


class VisualizationManager:
    """
    Central manager for all visualizations.
    Supports attention maps, layer analysis, gradients, embeddings, etc.
    """
    
    def __init__(self):
        self.style = "seaborn-v0_8-darkgrid"
        self.colormap = "viridis"
        plt.style.use(self.style)
        sns.set_palette("husl")
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        title: Optional[str] = None,
        interactive: bool = True
    ) -> Any:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights tensor (seq_len, seq_len) or (batch, heads, seq, seq)
            tokens: Token labels
            layer_idx: Layer index
            head_idx: Head index
            title: Plot title
            interactive: Use Plotly (True) or matplotlib (False)
            
        Returns:
            Plotly figure or matplotlib figure
        """
        # Extract single attention matrix
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx, :, :].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head_idx, :, :].detach().cpu().numpy()
        else:
            attn = attention_weights.detach().cpu().numpy()
        
        if tokens is None:
            tokens = [f"Tok_{i}" for i in range(attn.shape[0])]
        
        if interactive:
            # Plotly interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title or f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}",
                xaxis_title="Key Tokens",
                yaxis_title="Query Tokens",
                width=800,
                height=800,
                hovermode='closest'
            )
            
            return fig
        else:
            # Matplotlib static heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_title(title or f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
            plt.colorbar(im, ax=ax, label='Attention Weight')
            plt.tight_layout()
            
            return fig
    
    def plot_attention_rollout(
        self,
        attention_maps: List[torch.Tensor],
        tokens: List[str],
        num_layers: int = 12
    ) -> go.Figure:
        """
        Create attention rollout visualization (gradient-style).
        
        Args:
            attention_maps: List of attention maps per layer
            tokens: Token labels
            num_layers: Number of layers to show
            
        Returns:
            Plotly figure
        """
        # Average attention across heads
        avg_attn = []
        for attn in attention_maps[:num_layers]:
            avg = attn.mean(dim=1) if attn.dim() == 4 else attn
            avg_attn.append(avg[0].detach().cpu().numpy())
        
        # Create subplots
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Layer {i}" for i in range(num_layers)])
        
        for i, attn in enumerate(avg_attn):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(z=attn, colorscale='Viridis', showscale=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Attention Rollout Across Layers",
            height=300 * rows,
            width=300 * cols,
            showlegend=False
        )
        
        return fig
    
    def plot_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0
    ) -> go.Figure:
        """
        Plot attention entropy per token.
        Higher entropy = more uniform attention distribution.
        """
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx, :, :].detach().cpu().numpy()
        else:
            attn = attention_weights.detach().cpu().numpy()
        
        epsilon = 1e-10
        entropy = -np.sum(attn * np.log(attn + epsilon), axis=-1)
        
        fig = go.Figure(data=go.Bar(
            x=list(range(len(entropy))),
            y=entropy,
            marker_color='#21908C',
            hovertemplate='Token %{x}<br>Entropy: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Attention Entropy - Layer {layer_idx}, Head {head_idx}",
            xaxis_title="Token Position",
            yaxis_title="Entropy (bits)",
            template='plotly_white'
        )
        
        return fig
    
    def plot_layer_activation_maps(
        self,
        hidden_states: List[torch.Tensor],
        layer_names: List[str]
    ) -> go.Figure:
        """
        Visualize layer activations as heatmaps.
        
        Args:
            hidden_states: List of hidden state tensors per layer
            layer_names: Names of layers
        """
        num_layers = len(hidden_states)
        
        fig = make_subplots(rows=num_layers, cols=1, subplot_titles=layer_names)
        
        for i, hs in enumerate(hidden_states):
            # Average over batch and head
            hs_np = hs.mean(dim=(0, 1)).detach().cpu().numpy() if hs.dim() > 2 else hs[0].detach().cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(
                    z=hs_np.reshape(1, -1),
                    colorscale='Viridis',
                    showscale=(i == num_layers - 1),
                    hovertemplate='Activation: %{z:.4f}<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Layer Activation Maps",
            height=300 * num_layers,
            width=800
        )
        
        return fig
    
    def plot_gradient_flow(
        self,
        model: torch.nn.Module,
        named_parameters: List[Tuple[str, torch.nn.Parameter]]
    ) -> go.Figure:
        """
        Visualize gradient flow through layers.
        """
        layer_grads = {}
        
        for name, param in named_parameters:
            if param.grad is not None:
                layer_name = name.split('.')[0] if '.' in name else name
                grad_norm = param.grad.norm().item()
                
                if layer_name not in layer_grads:
                    layer_grads[layer_name] = []
                layer_grads[layer_name].append(grad_norm)
        
        layers = list(layer_grads.keys())
        avg_grads = [np.mean(layer_grads[l]) for l in layers]
        
        fig = go.Figure(data=go.Bar(
            x=layers,
            y=avg_grads,
            marker_color=avg_grads,
            marker_colorscale='Viridis',
            hovertemplate='Layer: %{x}<br>Gradient Norm: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Gradient Flow Through Layers",
            xaxis_title="Layer",
            yaxis_title="Average Gradient Norm",
            template='plotly_white'
        )
        
        return fig
    
    def plot_weight_distributions(
        self,
        model: torch.nn.Module,
        layer_types: List[str] = ['linear', 'attention']
    ) -> go.Figure:
        """
        Visualize weight distributions for different layer types.
        """
        fig = make_subplots(rows=2, cols=2, subplot_titles=['Linear Weights', 'Attention Weights', 'Embeddings', 'LayerNorm'])
        
        linear_weights = []
        attn_weights = []
        embed_weights = []
        layernorm_weights = []
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'linear' in name or 'dense' in name:
                    linear_weights.append(param.detach().cpu().numpy().flatten())
                elif 'attention' in name.lower() or 'attn' in name:
                    attn_weights.append(param.detach().cpu().numpy().flatten())
                elif 'embed' in name:
                    embed_weights.append(param.detach().cpu().numpy().flatten())
                elif 'norm' in name:
                    layernorm_weights.append(param.detach().cpu().numpy().flatten())
        
        # Plot histograms
        if linear_weights:
            fig.add_trace(go.Histogram(x=np.concatenate(linear_weights), name='Linear'), row=1, col=1)
        if attn_weights:
            fig.add_trace(go.Histogram(x=np.concatenate(attn_weights), name='Attention'), row=1, col=2)
        if embed_weights:
            fig.add_trace(go.Histogram(x=np.concatenate(embed_weights), name='Embeddings'), row=2, col=1)
        if layernorm_weights:
            fig.add_trace(go.Histogram(x=np.concatenate(layernorm_weights), name='LayerNorm'), row=2, col=2)
        
        fig.update_layout(
            title="Weight Distributions by Layer Type",
            height=600,
            width=800,
            showlegend=True,
            barmode='overlay'
        )
        
        return fig
    
    def plot_embedding_projection(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = 'pca'
    ) -> go.Figure:
        """
        Project embeddings to 2D/3D using PCA or t-SNE.
        
        Args:
            embeddings: Embedding tensor (num_samples, dim)
            labels: Labels for coloring
            method: Projection method ('pca', 'tsne', 'umap')
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        embeddings_np = embeddings.detach().cpu().numpy()
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings_np)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings_np)
        else:
            reduced = embeddings_np[:, :2]
        
        fig = go.Figure(data=go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=labels if labels else 'blue',
                colorscale='Viridis',
                showscale=True
            ),
            text=labels,
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Label: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Embedding Projection ({method.upper()})",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template='plotly_white'
        )
        
        return fig
    
    def create_model_dashboard(
        self,
        model: torch.nn.Module,
        attention_maps: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        gradient_info: Any
    ) -> go.Figure:
        """
        Create comprehensive model dashboard.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Attention Heatmap', 'Attention Entropy', 'Layer Activations',
                'Gradient Flow', 'Weight Distribution', 'Model Summary'
            ],
            specs=[
                [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'table'}]
            ]
        )
        
        # Add subplots (simplified - full implementation would add actual data)
        
        fig.update_layout(
            title="Model Analysis Dashboard",
            height=800,
            width=1200,
            showlegend=False
        )
        
        return fig
    
    def fig_to_html(self, fig) -> str:
        """Convert plotly figure to HTML."""
        return fig.to_html(full_html=False)
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
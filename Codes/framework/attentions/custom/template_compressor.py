# framework/compression/custom/template_compressor.py
"""
Template for creating new compression methods.

To create a new compressor:
    1. Copy this file to your custom directory
    2. Rename the class and file
    3. Implement the compress() method
    4. Register with @COMPRESSOR_REGISTRY.register("your_name")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ...core.registry import COMPRESSOR_REGISTRY
from ...core.base import BaseCompressor


@COMPRESSOR_REGISTRY.register("your_compressor_name")
class YourCustomCompressor(BaseCompressor):
    """
    Your custom compression method.
    
    This template provides a starting point for implementing
    new compression techniques.
    
    Supported compression types:
        - Low-rank (SVD, CUR, TT)
        - Pruning (magnitude, gradient-based)
        - Quantization (uniform, non-uniform)
        - Hybrid (combination of methods)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        target_rank: int = 64,
        method: str = "standard",
        **kwargs
    ):
        """
        Initialize your custom compressor.
        
        Args:
            model: Model to compress
            target_sparsity: Target sparsity level (0-1)
            target_rank: Target rank for low-rank methods
            method: Compression method variant
            **kwargs: Additional custom parameters
        """
        super().__init__(model, name="your_compressor", **kwargs)
        self.target_sparsity = target_sparsity
        self.target_rank = target_rank
        self.method = method
        self.custom_param = kwargs.get('custom_param', 0.1)
        
        # Track compression statistics
        self.compressed_layers = []
        self.compression_stats = {}
    
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply compression to the model.
        
        Returns:
            Tuple of (compressed_model, compression_statistics)
        """
        # Apply compression to each layer
        self._compress_embeddings()
        self._compress_attention_layers()
        self._compress_ffn_layers()
        self._compress_classifier()
        
        # Update statistics
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = self._compute_stats()
        
        return self.model, self.get_compression_stats()
    
    def _compress_embeddings(self):
        """Compress embedding layers."""
        if hasattr(self.model, 'embeddings'):
            embedding_layer = self.model.embeddings.word_embeddings
            original_size = embedding_layer.weight.shape
            compressed_size = self._get_compressed_size(original_size)
            
            self._apply_compression_to_layer(
                embedding_layer,
                original_size,
                compressed_size,
                layer_name="embeddings"
            )
    
    def _compress_attention_layers(self):
        """Compress attention layers."""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                self._compress_linear_layers(module, name, layer_type="attention")
    
    def _compress_ffn_layers(self):
        """Compress FFN layers."""
        for name, module in self.model.named_modules():
            if 'ffn' in name.lower() or 'intermediate' in name.lower() or 'output' in name.lower():
                self._compress_linear_layers(module, name, layer_type="ffn")
    
    def _compress_classifier(self):
        """Compress classifier head."""
        if hasattr(self.model, 'classifier'):
            self._compress_linear_layers(self.model.classifier, "classifier", layer_type="classifier")
    
    def _compress_linear_layers(self, module: nn.Module, parent_name: str, layer_type: str):
        """Compress linear layers within a module."""
        for name, submodule in module.named_modules():
            if isinstance(submodule, nn.Linear):
                original_size = submodule.weight.shape
                compressed_size = self._get_compressed_size(original_size)
                
                # Create compressed version based on method
                if self.method == "low_rank":
                    new_layer = self._create_low_rank_layer(submodule, compressed_size)
                elif self.method == "pruning":
                    new_layer = self._create_pruned_layer(submodule, compressed_size)
                elif self.method == "quantization":
                    new_layer = self._create_quantized_layer(submodule, compressed_size)
                else:
                    new_layer = self._create_hybrid_layer(submodule, compressed_size)
                
                # Replace the layer
                full_name = f"{parent_name}.{name}" if name else parent_name
                self._replace_layer(submodule, new_layer, full_name)
                self.compressed_layers.append({
                    'name': full_name,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'type': layer_type,
                    'method': self.method
                })
    
    def _get_compressed_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Determine compressed size for a layer.
        
        Args:
            original_size: Original dimensions (out_features, in_features)
            
        Returns:
            Compressed dimensions (out_features, in_features)
        """
        out_features, in_features = original_size
        
        if self.method == "low_rank":
            # Keep same dimensions but use low-rank factorization
            return (out_features, in_features)
        elif self.method == "pruning":
            # Reduce dimensions proportionally
            new_out = max(1, int(out_features * (1 - self.target_sparsity)))
            new_in = max(1, int(in_features * (1 - self.target_sparsity)))
            return (new_out, new_in)
        else:
            return (out_features, in_features)
    
    def _create_low_rank_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> nn.Module:
        """Create low-rank version of a linear layer."""
        from ..svd_based import SVDLinearLayer
        rank = min(self.target_rank, min(original_layer.weight.shape))
        return SVDLinearLayer(original_layer, rank)
    
    def _create_pruned_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> nn.Module:
        """Create pruned version of a linear layer."""
        new_out, new_in = compressed_size
        new_layer = nn.Linear(new_in, new_out)
        
        # Copy weights from original (prune by taking top k)
        with torch.no_grad():
            weight = original_layer.weight.data
            # Keep top rows by L2 norm
            row_norms = torch.norm(weight, dim=1)
            _, indices = torch.topk(row_norms, new_out)
            new_layer.weight.data = weight[indices, :new_in]
            
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data[indices]
        
        return new_layer
    
    def _create_quantized_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> nn.Module:
        """Create quantized version of a linear layer."""
        bits = self.custom_param if isinstance(self.custom_param, int) else 8
        scale = (2 ** (bits - 1)) - 1
        
        with torch.no_grad():
            weight = original_layer.weight.data
            weight_quant = torch.clamp(weight * scale, -scale, scale)
            weight_quant = torch.round(weight_quant) / scale
        
        new_layer = nn.Linear(original_layer.in_features, original_layer.out_features)
        new_layer.weight.data = weight_quant
        if original_layer.bias is not None:
            new_layer.bias.data = original_layer.bias.data
        
        return new_layer
    
    def _create_hybrid_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> nn.Module:
        """Create hybrid compression (low-rank + pruning)."""
        # First prune
        pruned_layer = self._create_pruned_layer(original_layer, compressed_size)
        # Then apply low-rank
        rank = min(self.target_rank // 2, min(pruned_layer.weight.shape))
        return self._create_low_rank_layer(pruned_layer, (rank, rank))
    
    def _replace_layer(self, old_layer: nn.Module, new_layer: nn.Module, layer_name: str):
        """Replace a layer in the model."""
        # Find parent module
        parts = layer_name.split('.')
        if len(parts) == 1:
            parent = self.model
            child_name = parts[0]
        else:
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    return
            child_name = parts[-1]
        
        if hasattr(parent, child_name):
            setattr(parent, child_name, new_layer)
    
    def _compute_stats(self) -> Dict[str, Any]:
        """Compute compression statistics."""
        total_original = self._original_params
        total_compressed = self._compressed_params
        
        return {
            'total_layers_compressed': len(self.compressed_layers),
            'compression_ratio': total_compressed / total_original if total_original > 0 else 1.0,
            'parameters_saved': total_original - total_compressed,
            'reduction_percentage': (1 - total_compressed / total_original) * 100 if total_original > 0 else 0,
            'compressed_layers': self.compressed_layers,
            'compression_method': self.method,
        }
    
    def get_layer_wise_report(self) -> pd.DataFrame:
        """Get detailed layer-wise compression report."""
        import pandas as pd
        return pd.DataFrame(self.compressed_layers)
    
    def visualize_compression(self, save_path: Optional[str] = None):
        """Visualize compression results."""
        import matplotlib.pyplot as plt
        
        if not self.compressed_layers:
            print("No compression statistics available")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Parameter reduction by layer
        layers = [l['name'].split('.')[-1][:20] for l in self.compressed_layers]
        original_sizes = [l['original_size'][0] * l['original_size'][1] for l in self.compressed_layers]
        compressed_sizes = [l['compressed_size'][0] * l['compressed_size'][1] for l in self.compressed_layers]
        
        x = range(len(layers))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], original_sizes, width, label='Original', alpha=0.7)
        axes[0].bar([i + width/2 for i in x], compressed_sizes, width, label='Compressed', alpha=0.7)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Parameters')
        axes[0].set_title('Parameter Reduction per Layer')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0].legend()
        
        # Compression ratio by type
        types = {}
        for l in self.compressed_layers:
            t = l['type']
            types[t] = types.get(t, 0) + 1
        
        axes[1].pie(types.values(), labels=types.keys(), autopct='%1.1f%%')
        axes[1].set_title('Compression Distribution by Layer Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
"""
Template for creating new compression methods.
Ready to use – copy and customize the compress flow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from framework.core.base import BaseCompressor
import torch.nn.functional as F

# Simple low‑rank linear layer used for demonstration; you can replace with your own.
class SVDLinearLayer(nn.Module):
    """Low‑rank (U @ diag(S) @ V) layer created from an existing Linear."""
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        out_features, in_features = original_layer.weight.shape
        self.rank = rank
        W = original_layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        self.U = nn.Linear(rank, out_features, bias=False)
        self.V = nn.Linear(in_features, rank, bias=False)
        self.U.weight.data = (U[:, :rank] * S[:rank].unsqueeze(0)).float()
        self.V.weight.data = Vt[:rank, :].float()
        self.bias = None
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())

    def forward(self, x):
        x = self.V(x)
        x = self.U(x)
        if self.bias is not None:
            x = x + self.bias
        return x


@COMPRESSOR_REGISTRY.register("your_compressor_name")
class YourCustomCompressor(BaseCompressor):
    """
    Your custom compression method.
    
    This template supports:
        - Low‑rank (SVD)
        - Pruning (top‑k by row norm)
        - Quantization (uniform)
        - Hybrid (prune + low‑rank)
    
    Simply change `self.method` and adjust parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        target_rank: int = 64,
        method: str = "standard",
        **kwargs
    ):
        super().__init__(model, name="your_compressor", **kwargs)
        self.target_sparsity = target_sparsity
        self.target_rank = target_rank
        self.method = method
        self.custom_param = kwargs.get('custom_param', 0.1)   # e.g., bits for quantization

        self.compressed_layers = []
        self.compression_stats = {}

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply compression and return (compressed_model, stats)."""
        self._compress_embeddings()
        self._compress_attention_layers()
        self._compress_ffn_layers()
        self._compress_classifier()

        self._compressed_params = self._count_params(self.model)
        self._compression_stats = self._compute_stats()
        return self.model, self.get_compression_stats()

    def _compress_embeddings(self):
        if hasattr(self.model, 'embeddings'):
            emb_layer = self.model.embeddings.word_embeddings
            self._compress_linear_layers(emb_layer, "embeddings", "embedding")

    def _compress_attention_layers(self):
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                self._compress_linear_layers(module, name, "attention")

    def _compress_ffn_layers(self):
        for name, module in self.model.named_modules():
            if 'ffn' in name.lower() or 'intermediate' in name.lower() or 'output' in name.lower():
                self._compress_linear_layers(module, name, "ffn")

    def _compress_classifier(self):
        if hasattr(self.model, 'classifier'):
            self._compress_linear_layers(self.model.classifier, "classifier", "classifier")

    def _compress_linear_layers(self, module: nn.Module, parent_name: str, layer_type: str):
        for child_name, child in module.named_modules():
            if isinstance(child, nn.Linear):
                original_size = child.weight.shape
                compressed_size = self._get_compressed_size(original_size)

                if self.method == "low_rank":
                    new_layer = SVDLinearLayer(child, self.target_rank)
                elif self.method == "pruning":
                    new_layer = self._create_pruned_layer(child, compressed_size)
                elif self.method == "quantization":
                    new_layer = self._create_quantized_layer(child)
                else:
                    new_layer = self._create_hybrid_layer(child, compressed_size)

                full_name = f"{parent_name}.{child_name}" if child_name else parent_name
                self._replace_layer(module, child_name, new_layer, full_name)
                self.compressed_layers.append({
                    'name': full_name,
                    'original_size': original_size,
                    'compressed_size': compressed_size if not isinstance(new_layer, SVDLinearLayer) else (new_layer.V.out_features, new_layer.U.in_features),
                    'type': layer_type,
                    'method': self.method
                })

    def _get_compressed_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        out_f, in_f = original_size
        if self.method == "pruning":
            new_out = max(1, int(out_f * (1 - self.target_sparsity)))
            new_in = max(1, int(in_f * (1 - self.target_sparsity)))
            return (new_out, new_in)
        return original_size

    def _create_pruned_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> nn.Linear:
        new_out, new_in = compressed_size
        new_layer = nn.Linear(new_in, new_out)
        with torch.no_grad():
            weight = original_layer.weight.data
            row_norms = torch.norm(weight, dim=1)
            _, indices = torch.topk(row_norms, new_out)
            new_layer.weight.data = weight[indices][:, :new_in]
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data[indices]
        return new_layer

    def _create_quantized_layer(self, original_layer: nn.Linear) -> nn.Linear:
        bits = int(self.custom_param) if isinstance(self.custom_param, (int, float)) else 8
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

    def _create_hybrid_layer(self, original_layer: nn.Linear, compressed_size: Tuple[int, int]) -> SVDLinearLayer:
        # first prune, then low‑rank
        pruned = self._create_pruned_layer(original_layer, compressed_size)
        rank = min(self.target_rank // 2, min(pruned.weight.shape))
        return SVDLinearLayer(pruned, rank)

    def _replace_layer(self, parent_module: nn.Module, child_name: str, new_layer: nn.Module, full_name: str):
        if hasattr(parent_module, child_name):
            setattr(parent_module, child_name, new_layer)
        else:
            print(f"Warning: could not find {child_name} in {full_name}")

    def _compute_stats(self) -> Dict[str, Any]:
        return {
            'total_layers_compressed': len(self.compressed_layers),
            'compression_ratio': self._compressed_params / self._original_params if self._original_params > 0 else 1.0,
            'parameters_saved': self._original_params - self._compressed_params,
            'reduction_percentage': (1 - self._compressed_params / self._original_params) * 100 if self._original_params > 0 else 0,
            'compressed_layers': self.compressed_layers,
            'compression_method': self.method,
        }

    def get_layer_wise_report(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self.compressed_layers)

    def visualize_compression(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        if not self.compressed_layers:
            print("No compression statistics available")
            return

        layers = [l['name'].split('.')[-1][:20] for l in self.compressed_layers]
        orig_params = [l['original_size'][0]*l['original_size'][1] for l in self.compressed_layers]
        comp_params = [l['compressed_size'][0]*l['compressed_size'][1] for l in self.compressed_layers]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = range(len(layers))
        width = 0.35
        axes[0].bar([i - width/2 for i in x], orig_params, width, label='Original', alpha=0.7)
        axes[0].bar([i + width/2 for i in x], comp_params, width, label='Compressed', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0].legend()

        type_counts = {}
        for l in self.compressed_layers:
            t = l['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        axes[1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
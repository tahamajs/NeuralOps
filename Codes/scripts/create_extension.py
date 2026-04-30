#!/usr/bin/env python
# scripts/create_extension.py
"""
Helper script to create new framework extensions.
Run: python scripts/create_extension.py --type attention --name my_attention
"""

import os
import argparse
from pathlib import Path


def create_attention(name: str, output_dir: str = "framework/attentions/custom"):
    """Create a new attention template."""
    os.makedirs(output_dir, exist_ok=True)
    
    content = f'''"""
{name} - Custom attention mechanism for SVD-Lin framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.base import BaseAttention


@ATTENTION_REGISTRY.register("{name}")
class {name.title().replace('_', '')}Attention(BaseAttention):
    """
    {name.title()} Attention mechanism.
    
    TODO: Add description of your attention mechanism.
    """
    
    def __init__(self, config, rank: Optional[int] = None, **kwargs):
        super().__init__(config, rank)
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_probs_dropout_prob
        
        # TODO: Define your layers here
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # TODO: Implement your attention logic here
        
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)
        
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)
        
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        
        context = torch.matmul(probs, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        output = self.out_proj(context)
        
        if output_attentions:
            return output, probs
        return output, None
    
    @property
    def complexity(self) -> str:
        return "quadratic"
'''
    
    filepath = os.path.join(output_dir, f"{name}.py")
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Created attention: {filepath}")
    return filepath


def create_compressor(name: str, output_dir: str = "framework/compression/custom"):
    """Create a new compressor template."""
    os.makedirs(output_dir, exist_ok=True)
    
    content = f'''"""
{name} - Custom compression method for SVD-Lin framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.base import BaseCompressor


@COMPRESSOR_REGISTRY.register("{name}")
class {name.title().replace('_', '')}Compressor(BaseCompressor):
    """
    {name.title()} Compression method.
    
    TODO: Add description of your compression method.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, name="{name}", **kwargs)
        self.target_sparsity = kwargs.get('sparsity', 0.5)
        self.target_rank = kwargs.get('rank', 64)
    
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply compression to the model."""
        # TODO: Implement your compression logic here
        
        # Track compressed layers
        compressed_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Apply compression to linear layer
                # This is a placeholder
                compressed_count += 1
        
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = {
            "compressed_layers": compressed_count,
            "method": self.name,
        }
        
        return self.model, self.get_compression_stats()
'''
    
    filepath = os.path.join(output_dir, f"{name}.py")
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Created compressor: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Create framework extensions")
    parser.add_argument("--type", "-t", required=True, choices=["attention", "compressor", "visualizer", "trainer"],
                        help="Type of extension to create")
    parser.add_argument("--name", "-n", required=True, help="Name of the extension")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    if args.type == "attention":
        output_dir = args.output or "framework/attentions/custom"
        create_attention(args.name, output_dir)
    elif args.type == "compressor":
        output_dir = args.output or "framework/compression/custom"
        create_compressor(args.name, output_dir)
    else:
        print(f"Extension type '{args.type}' template not yet implemented")
        print("Check back soon!")


if __name__ == "__main__":
    main()
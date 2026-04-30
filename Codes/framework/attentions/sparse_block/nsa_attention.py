# framework/attentions/nsa_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("nsa")
class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention (NSA): Hardware-aligned and natively trainable sparse attention.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        # Will leverage internal NSA kernels
        pass
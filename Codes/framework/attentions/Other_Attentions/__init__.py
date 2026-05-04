"""
Attention implementations for the Efficient Transformer Toolkit.
All classes are automatically registered into ATTENTION_REGISTRY.
"""

from .full_attention import FullAttention
from .svd_attention import SVDLinearAttention
from .linformer_attention import LinformerAttention
from .nexus import NEXUSAttention, NEXUSConfig, inject_nexus
from .compressed_attention import CompressedAttentionLayer, CompressedAttentionLayer2

# The registry is populated by the decorators inside each module.
# Importing the modules triggers registration.
# No need to manually add to registry here.

__all__ = [
    "FullAttention",
    "SVDLinearAttention",
    "LinformerAttention",
    "NEXUSAttention",
    "NEXUSConfig",
    "CompressedAttentionLayer",
    "CompressedAttentionLayer2",
]
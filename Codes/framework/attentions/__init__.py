# studio/attentions/__init__.py
"""
Attention mechanisms for SVD‑Lin Studio.
"""

from .registry import ATTENTION_REGISTRY

# Import all attention classes so they get registered
from .full_attention import FullAttention
from .svd_attention import SVDLinearAttention
from .linformer_attention import LinformerAttention
from .compressed_attention import CompressedAttentionLayer, CompressedAttentionLayer2

# (Add other attention files as you add them)

__all__ = [
    "ATTENTION_REGISTRY",
    "FullAttention",
    "SVDLinearAttention",
    "LinformerAttention",
    "CompressedAttentionLayer",
    "CompressedAttentionLayer2",
]
# framework/attentions/specialized/__init__.py
"""
Specialized attention mechanisms (sparse, linear, clustering, LSH, etc.).
All modules are automatically registered in ATTENTION_REGISTRY.
"""

from . import adasplash_attention
from . import adaptive_span_attention
from . import block_transformer_hiba
from . import cluster_attention
from . import flash_attention3
from . import flash_attention3_fp8
from . import full_attention
from . import hamming_attention
from . import hla_attention
from . import infini_attention
from . import linformer_attention
from . import lsh_attention
from . import mixattention_moska
from . import moba_attention
from . import nha_attention
from . import nsa_attention
from . import routing_attention
from . import span_expanded_attention
from . import vig_attention
from . import xattention

# Optionally re‑export classes for direct import
try:
    from .adasplash_attention import AdaSplashAttention
except ImportError:
    AdaSplashAttention = None
try:
    from .adaptive_span_attention import AdaptiveSpanAttention
except ImportError:
    AdaptiveSpanAttention = None
try:
    from .block_transformer_hiba import BlockTransformerHIBA
except ImportError:
    BlockTransformerHIBA = None
try:
    from .cluster_attention import ClusterAttention
except ImportError:
    ClusterAttention = None
try:
    from .flash_attention3 import FlashAttention3
except ImportError:
    FlashAttention3 = None
try:
    from .flash_attention3_fp8 import FlashAttention3FP8
except ImportError:
    FlashAttention3FP8 = None
from .full_attention import FullAttention
from .hamming_attention import HammingAttention
from .hla_attention import HLAAttention
from .infini_attention import InfiniAttention
from .linformer_attention import LinformerAttention
from .lsh_attention import LSHAttention
from .mixattention_moska import MixAttentionMoSKA
from .moba_attention import MoBAAttention
from .nha_attention import NHAAttention
from .nsa_attention import NativeSparseAttention
from .routing_attention import RoutingAttention
from .span_expanded_attention import SpanExpandedAttention
from .vig_attention import ViGAttention
from .xattention import XAttention

__all__ = [
    "AdaSplashAttention",
    "AdaptiveSpanAttention",
    "BlockTransformerHIBA",
    "ClusterAttention",
    "FlashAttention3",
    "FlashAttention3FP8",
    "FullAttention",
    "HammingAttention",
    "HLAAttention",
    "InfiniAttention",
    "LinformerAttention",
    "LSHAttention",
    "MixAttentionMoSKA",
    "MoBAAttention",
    "NHAAttention",
    "NativeSparseAttention",
    "RoutingAttention",
    "SpanExpandedAttention",
    "ViGAttention",
    "XAttention",
]
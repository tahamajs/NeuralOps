from .config import HybridLocalKernelConfig
from .linear import FactorisedLinear
from .attention import _local_window_attention_v2, _global_linear_attention_v2
from .layer import (
    HybridLocalKernelAttention,
    inject_hybrid_local_kernel,
)

# Aliases for backward compatibility
AegisConfigV2 = HybridLocalKernelConfig
AegisAttentionV1 = HybridLocalKernelAttention
AegisAttentionV2 = HybridLocalKernelAttention
AegisAttentionV3 = HybridLocalKernelAttention
AegisAttentionV4 = HybridLocalKernelAttention
AegisAttentionV5 = HybridLocalKernelAttention

__all__ = [
    "HybridLocalKernelConfig",
    "FactorisedLinear",
    "_local_window_attention_v2",
    "_global_linear_attention_v2",
    "HybridLocalKernelAttention",
    "inject_hybrid_local_kernel",
    "AegisConfigV2",
    "AegisAttentionV1",
    "AegisAttentionV2",
    "AegisAttentionV3",
    "AegisAttentionV4",
    "AegisAttentionV5",
]

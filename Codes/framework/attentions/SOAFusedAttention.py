"""Shim for soafusedattention re-exporting from SOAFusedAttention."""
from framework.attentions.SOAFusedAttention import (
    SOAFusedAttention,
    inject_soa_fused_attn,
    inject_soa_fused_module,
    inject_soafusedattention,
)

__all__ = [
    "SOAFusedAttention",
    "inject_soa_fused_attn",
    "inject_soa_fused_module",
    "inject_soafusedattention",
]

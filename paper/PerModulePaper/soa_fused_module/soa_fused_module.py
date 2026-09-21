"""Shim for soa_fused_module re-exporting from SOAFusedAttention."""
from framework.attentions.SOAFusedAttention import (
    SOAFusedAttention,
    SOAFusedAttention as SOAFusedModule,
    inject_soa_fused_attn,
    inject_soa_fused_module,
    inject_soafusedattention,
)

__all__ = [
    "SOAFusedAttention",
    "SOAFusedModule",
    "inject_soa_fused_attn",
    "inject_soa_fused_module",
    "inject_soafusedattention",
]

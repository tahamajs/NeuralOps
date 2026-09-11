"""Shim for soa_module re-exporting from soa_attn."""
from framework.attentions.soa_attn import (
    SOAAttention,
    SOAAttention as SOAModule,
    inject_soa_attn,
    inject_soa,
    inject_soa_module,
)

__all__ = [
    "SOAAttention",
    "SOAModule",
    "inject_soa_attn",
    "inject_soa",
    "inject_soa_module",
]

r"""
Random Projection Compressed (RPC) Attention.
Re-exports from framework.attentions.rpc_attention.
"""

from framework.attentions.rpc_attention import RPCAttention, LoRALinear, inject_rpc

__all__ = ["RPCAttention", "LoRALinear", "inject_rpc"]

r"""
Token-Compressed Attention Wrapper / Alias (tt.py).

Exposes TokenCompressedAttention and injection helpers from token_compressed_attention.
"""

from framework.attentions.token_compressed_attention import (
    TokenCompressedAttention,
    inject_token_compressed_attn,
)

inject_token_compressed = inject_token_compressed_attn

__all__ = ["TokenCompressedAttention", "inject_token_compressed_attn", "inject_token_compressed"]


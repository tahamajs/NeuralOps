"""
Token-Compressed Attention.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

from framework.attentions.token_compressed_attention import (
    TokenCompressedAttention,
    inject_token_compressed_attn,
)

__all__ = ["TokenCompressedAttention", "inject_token_compressed_attn"]

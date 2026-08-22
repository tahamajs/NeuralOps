r"""
SVD-LLM V2 Attention Alias.
Maps to `framework.attentions.svd_llm_v2_attention.SVDLLMV2Attention`.
"""

from framework.attentions.svd_llm_v2_attention import (
    SVDLLMV2Attention,
    inject_svd_llm_v2,
)

__all__ = ["SVDLLMV2Attention", "inject_svd_llm_v2"]

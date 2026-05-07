"""Full (vanilla) self‑attention – registers as 'full'."""

from transformers.models.bert.modeling_bert import BertSelfAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("full")
class FullAttention(BertSelfAttention):
    """
    Standard BERT self‑attention.
    This class simply inherits from BertSelfAttention.
    """
    pass
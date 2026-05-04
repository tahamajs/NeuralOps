# framework/attentions/specialized/full_attention.py
from transformers.models.bert.modeling_bert import BertSelfAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("full")
class FullAttention(BertSelfAttention):
    """
    Standard BERT self‑attention (baseline).
    """
    pass
# framework/soa/__init__.py
"""
SOA (SVD-Optimized Architecture) – Low‑rank transformer models.

This module provides SVD‑based compressed transformers with a shared basis
computed from BERT's QKV weights. It supports:
- Computing a global basis from a trained BERT model.
- Migrating BERT weights to SOA format.
- Fine‑tuning and exporting to ONNX/CoreML.

Example:
    >>> from framework.soa import SOAConfig, SOABert, migrate_bert_to_soa
    >>> from transformers import BertModel
    >>> 
    >>> bert = BertModel.from_pretrained("bert-base-uncased")
    >>> basis = compute_global_basis_from_hidden_states(bert, rank=64)
    >>> soa = SOABert(SOAConfig(k_rank=64))
    >>> soa = migrate_bert_to_soa(bert, soa, basis)
"""

from .config import SOAConfig
from .attention import SOAFusedAttention
from .model import SOABlock, SOABert, SVDBasedBERT, SVDBasedBlock
from .svd_migration import (
    compute_global_basis,
    compute_global_basis_from_hidden_states,
    compute_global_basis_from_input_states,
    migrate_bert_to_soa,
)
from .utils import set_seed, get_device, count_parameters
from .export_onnx import export_onnx
from .export_coreml import export_to_coreml

__all__ = [
    "SOAConfig",
    "SOAFusedAttention",
    "SOABlock",
    "SOABert",
    "SVDBasedBERT",
    "SVDBasedBlock",
    "compute_global_basis",
    "compute_global_basis_from_hidden_states",
    "compute_global_basis_from_input_states",
    "migrate_bert_to_soa",
    "set_seed",
    "get_device",
    "count_parameters",
    "export_onnx",
    "export_to_coreml",
]
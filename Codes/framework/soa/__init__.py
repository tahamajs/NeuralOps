# framework/soa/__init__.py
"""
SOA (SVD-Optimized Architecture) - Low-rank transformer models.

This module provides SVD-optimized transformer blocks that use a shared
low-rank basis for attention projections, significantly reducing parameters
while maintaining performance.

Components:
    - SOAConfig: Configuration for SOA models
    - SOAFusedAttention: Fused attention with low-rank projections
    - SOABlock: Transformer block with SOA attention
    - SOABert: Complete BERT model with SOA architecture
    - Migration utilities: Convert BERT to SOA format

Example:
    >>> from framework.soa import SOAConfig, SOABert, migrate_bert_to_soa
    >>> 
    >>> config = SOAConfig(k_rank=64)
    >>> soa_model = SOABert(config)
    >>> # Migrate from pretrained BERT
    >>> soa_model = migrate_bert_to_soa(bert_model, soa_model, basis)
"""

from .config import SOAConfig
from .attention import SOAFusedAttention
from .block import SOABlock
from .model import SOABert
from .svd_compression import compress_bert_true_svd, svd_compress_input, apply_svd_to_inputs
from .migration import compute_global_basis_from_bert, migrate_bert_to_soa

__all__ = [
    "SOAConfig",
    "SOAFusedAttention",
    "SOABlock",
    "SOABert",
    "compress_bert_true_svd",
    "svd_compress_input",
    "apply_svd_to_inputs",
    "compute_global_basis_from_bert",
    "migrate_bert_to_soa",
]

__version__ = "1.0.0"
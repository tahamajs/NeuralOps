# Adaptive Sparse Attention Module (adaptive_sparse_attention)

**Registry Keys**: `adaptive_sparse_attn`, `adaptive_sparse_attention`, `adaptive_sparse_inject`, `adaptive_sparse`
**Source**: `Codes/framework/attentions/adaptive_sparse_attention.py` <!-- source: Codes/framework/attentions/adaptive_sparse_attention.py lines 24, 169-170 -->
**Type**: Dynamic Top-k and Threshold-Based Adaptive Attention Sparsification
**Base Class**: `AdaptiveSparseAttention(BaseAttention)` <!-- source: Codes/framework/attentions/adaptive_sparse_attention.py line 169 -->

## Abstract
Transformer self-attention typically calculates a dense $L \times L$ score matrix, even though the vast majority of cross-token interactions carry near-zero probability mass. `adaptive_sparse_attention` implements dynamic top-$k$ and entropy-guided threshold sparsification to eliminate irrelevant attention connections. Operating across encoder (DistilBERT, RoBERTa) and decoder (GPT-2, Qwen 3.5) architectures, it applies model-aware sparsity defaults (10% for bidirectional encoders, 50% for causal decoders) with LoRA adapter recovery. Evaluated on SST-2, it achieves 88.99% accuracy on DistilBERT and 79.36% on GPT-2.

# AEGIS-Attention V3: Dynamic Context-Gated Hybrid Attention (aegis_v3)

**Registry Keys**: `aegis_v3`, `aegis_v3_inject`, `aegis_v3_fn`, `AegisAttention`, `aegis_v3_module` <!-- source: Codes/framework/attentions/aegis_v3.py lines 19, 90-95 -->
**Source**: `Codes/framework/attentions/aegis_v3.py` (789 lines)
**Type**: Hybrid Attention with Learnable Context Gating and Sliding-Window Softmax
**Base Class**: `AegisAttentionV3(BaseAttention)`

## Abstract
`aegis_v3` refines the arbitration mechanism between local sliding-window attention and global linear attention via a learnable token-level context gating network. Operating across DistilBERT, RoBERTa, and GPT-2 on SST-2, it achieves 78.44% accuracy on DistilBERT with peak VRAM of only 208.28 MB and throughput exceeding 12,500 tokens/sec.

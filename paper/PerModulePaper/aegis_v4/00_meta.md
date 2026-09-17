# AEGIS-Attention V4: Multi-Scale Sliding Window Attention (aegis_v4)

**Registry Keys**: `aegis_v4`, `aegis_v4_inject`, `aegis_v4_fn`, `aegis_v4_module`, `AegisAttention` <!-- source: Codes/framework/attentions/aegis_v4.py lines 19, 90-95 -->
**Source**: `Codes/framework/attentions/aegis_v4.py` (794 lines)
**Type**: Multi-Scale Sliding Window Attention with Hierarchical Context Bands
**Base Class**: `AegisAttentionV4(BaseAttention)`

## Abstract
`aegis_v4` extends the AEGIS architecture with hierarchical multi-scale sliding windows across attention heads ($W \in \{32, 64, 128\}$), providing diverse receptive field granularities within a single layer. Across DistilBERT, RoBERTa, and GPT-2 on SST-2, it achieves strong performance: 85.21% accuracy on DistilBERT with peak VRAM of only 193.86 MB (throughput 13,166 tokens/sec) and 70.99% accuracy on GPT-2.

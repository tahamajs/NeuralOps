# AEGIS-Attention V6: Unified Hybrid Attention with Physical Compression (aegis_v6)

**Registry Keys**: `aegis_v6`, `inject_aegis_v6`, `aegis_v6_inject`, `aegis_v6_attn` <!-- source: Codes/framework/attentions/aegis_v6.py line 19 -->
**Source**: `Codes/framework/attentions/aegis_v6.py` (892 lines)
**Type**: SVD Factorized Low-Rank Projections with Unified Sliding Window and Linear Attention
**Base Class**: `AegisAttentionV6(BaseAttention)`

## Abstract
`aegis_v6` delivers the apex implementation of the AEGIS attention series, unifying physical low-rank matrix compression with sliding-window dense attention and adaptive spectral linear kernel attention. It completely eliminates parameter inflation by utilizing true structural `FactorisedLowRankProjection` layers ($r=128$). Across standard backbones on SST-2, it achieves 87.04% accuracy on DistilBERT with a 20.17% parameter reduction (saving 13.5M parameters) and peak VRAM of 301.26 MB, 59.98% accuracy on RoBERTa with a 15.98% parameter reduction, and a 21.71% parameter reduction on GPT-2 (saving 27.0M parameters).

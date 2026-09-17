# Aegis-Hybrid Attention Mechanism (aegis_hybrid)

**Registry Keys**: `aegis_hybrid`, `aegis_hybrid_inject`, `inject_aegis_hybrid`, `aegis_hybrid_attn`, `aegis_hybrid_attention` <!-- source: Codes/framework/attentions/aegis_hybrid.py line 21 -->
**Source**: `Codes/framework/attentions/aegis_hybrid.py` (1037 lines)
**Type**: Multi-Branch Hybrid Attention with Sliding Window, Global Factorization, and Learned Gating
**Base Class**: `AegisHybridAttention(BaseAttention)`

## Abstract
`aegis_hybrid` fuses local sliding-window dense attention with global low-rank factorized attention and learned gating fusion. Projections are parameterized via `FactorizedLinear` layers ($x 	o 	ext{lin}_2(	ext{lin}_1(x))$ with rank $r=128$) combined with trainable residual LoRA adapters ($r_{	ext{lora}}=32$). Across DistilBERT, RoBERTa, and GPT-2, it achieves verified parameter reduction (10.9% on DistilBERT, 12.7% on RoBERTa, 18.4% on GPT-2) while delivering 84.98% SST-2 accuracy on DistilBERT.

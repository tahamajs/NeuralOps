# AEGIS-Attention V2: Parameter-Efficient Low-Rank Hybrid Attention (aegis_v2)

**Registry Keys**: `aegis_v2`, `aegis_v2_inject`, `inject_aegis_v2` <!-- source: Codes/framework/attentions/aegis_v2.py line 20 -->
**Source**: `Codes/framework/attentions/aegis_v2.py` (772 lines)
**Type**: True Parameter-Reducing Low-Rank Projection Factorization with Sliding Window and Gated Context
**Base Class**: `AegisAttentionV2(BaseAttention)`

## Abstract
`aegis_v2` introduces physical parameter reduction into the AEGIS hybrid architecture by replacing dense projection matrices with factorized lower-rank tensors ($r=128$) without parameter inflation. Combined with sliding-window local attention ($W=128$) and global linear kernel attention, it achieves a 17.07% parameter reduction on DistilBERT (dropping from 66.95M to 55.52M parameters) with 72.36% SST-2 accuracy and a 12.66% parameter reduction on RoBERTa with 62.39% accuracy.

# AEGIS-Hybrid Attention: Local Sliding-Window + Global Linear Attention

## 1. Mathematical Formulation & Architecture
Combines low-rank factorized projections ($W \approx W_1 W_2$), local windowed dense attention, token-pruning global linear attention, and learned adaptive gating:
$$O_i = \alpha_i \odot O^{\text{local}}_i + (1 - \alpha_i) \odot O^{\text{global}}_i$$
where $\alpha_i = \sigma(\operatorname{FusionGate}(X_i) + 1.8)$.

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: When invoked as `aegis_hybrid_attn`, direct class initialization did not invoke `_svd_init`, leaving projection weights random and hitting 50.92%.
- **Architectural Repair**: Unified class `__init__` to automatically invoke SVD weight transfer whenever `pretrained_attn` is present, added `from_distilbert_pretrained`, and calibrated gate bias to $+1.8$.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.94%** (previously 50.92% without injection hook).\n
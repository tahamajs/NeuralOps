# EAT: Efficient Attention Transformer with Progressive Token Pruning

## 1. Mathematical Formulation & Architecture
1. **Multi-Head Self-Attention**:
   $$O = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V W_O$$

2. **Progressive Token Pruning**:
   Selects top-$k$ informative tokens dynamically:
   $$X^{\text{pruned}} = X[:, \mathcal{I}_{\text{keep}}, :], \quad |\mathcal{I}_{\text{keep}}| = \lfloor \gamma \cdot L \rfloor$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Block-level wrapper duplicated LayerNorms and FFNs with random weights, destroying pretrained features and hitting 50.92%.
- **Architectural Repair**: Refactored EAT directly into the attention module interface with `from_distilbert_pretrained` and zero-shot weight inheritance.

## 3. Empirical Performance
- **SST-2 Accuracy**: **84.06%** (restored from 50.92%).\n
# GoldFinch-AHN: Adaptive Hybrid Network Combining Sliding Window and Gated DeltaNet

## 1. Mathematical Formulation & Architecture
1. **Multi-Head Sliding Window Attention**:
   $$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d_h}} + M_{\text{window}}, \quad O^{\text{short}} = \operatorname{Softmax}(S) V$$

2. **Long-Term Gated DeltaNet Memory**:
   $$O^{\text{long}} = \operatorname{DeltaNet}(X)$$

3. **Adaptive Gated Blending**:
   $$g = \sigma(X W_{\text{gate}} + 1.8)$$
   $$\text{Output} = W_O (g \odot O^{\text{short}} + (1 - g) \odot O^{\text{long}})$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Single-head dot-product over 768 dimensions destroyed multi-head subspace representations, and causal masking corrupted bidirectional token visibility (50.92%).
- **Architectural Repair**: Upgraded to 12-head attention, bidirectional sliding window, `from_distilbert_pretrained`, and positive gate bias.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.72%** (restored from 50.92%).\n
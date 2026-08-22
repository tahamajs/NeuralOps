# HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention

## 1. Mathematical Formulation & Architecture
1. **Rotary Position Embeddings (RoPE)**:
   $$Q_{\text{rope}} = \operatorname{RoPE}(Q), \quad K_{\text{rope}} = \operatorname{RoPE}(K)$$

2. **ELU+1 Linear Kernel Representation**:
   $$\phi(x) = \operatorname{ELU}(x) + 1$$
   $$O^{\text{linear}} = \frac{\phi(Q_{\text{rope}}) (\phi(K_{\text{rope}})^T V)}{\phi(Q_{\text{rope}}) (\sum \phi(K_{\text{rope}})^T) + \epsilon}$$

3. **Adaptive Gated Blending**:
   $$g = \sigma(X W_{\text{gate}} + 1.8)$$
   $$\text{Output} = W_O (g \odot O^{\text{dense}} + (1 - g) \odot O^{\text{linear}})$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Hard ReLU zeroed out 50% of activation projections, killing gradients; random initialization destroyed representations (50.92%).
- **Architectural Repair**: Added `from_distilbert_pretrained`, ELU+1 non-negative kernel, and adaptive dense-stream gating.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.83%** (restored from 50.92%).\n
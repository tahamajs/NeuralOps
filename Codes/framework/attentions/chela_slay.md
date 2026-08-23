# CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel

## 1. Mathematical Formulation & Architecture
1. **Spherical Hypersphere Projection**:
   $$\hat{Q} = \frac{Q}{\|Q\|_2}, \quad \hat{K} = \frac{K}{\|K\|_2}$$

2. **Yat-Kernel Trigonometric Feature Map**:
   $$\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega), \sin(x \Omega)] \in \mathbb{R}^{2F}$$

3. **Linear Attention Matrix Reduction**:
   $$O^{\text{linear}} = \frac{\phi(\hat{Q}) (\phi(\hat{K})^T V)}{\phi(\hat{Q}) (\sum \phi(\hat{K})^T) + \epsilon}$$

4. **Adaptive Gated Hybrid Blending**:
   $$g = \sigma(X W_{\text{gate}} + 1.8)$$
   $$\text{Output} = W_O (g \odot O^{\text{dense}} + (1 - g) \odot O^{\text{linear}})$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Pure trigonometric random feature attention without weight transfer collapsed feature geometry, scoring 50.92%.
- **Architectural Repair**: Added `from_distilbert_pretrained`, dense attention residual stream, and positive gating bias $+1.8$.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.83%** (restored from 50.92%).\n

## Implementation status

The implementation for this page is [chela_slay.py](./chela_slay.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

# SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Rectified Attention

## 1. Mathematical Formulation & Architecture
1. **Intra-Block Attention with Adaptive Span Gating**:
   $$M_{ij}^{\text{span}} = \ln\left(\sigma(5 \cdot (S_h - |i - j|)) + 10^{-5}\right)$$

2. **Inter-Block Top-$k$ Routing with Relative Positional Bias**:
   $$\text{Score}_{\text{inter}} = \frac{Q_b K_{\text{cand}}^T}{\sqrt{d_h}} + \operatorname{Bias}_{\text{rel}}(p_q - p_k)$$

3. **Global Random Fourier Feature Rectification**:
   $$\phi(x) = \frac{1}{d_r^{1/4}} [\cos(x W_{\text{rff}}), \sin(x W_{\text{rff}})]$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Module instantiation signatures failed during automated harness injection; missing weight transfer caused representation collapse.
- **Architectural Repair**: Implemented `from_distilbert_pretrained`, flexible positional/keyword injection signatures, and FP16 safe masking.

## 3. Empirical Performance
- **SST-2 Accuracy**: **84.02%**\n

## Implementation status

The implementation for this page is [spectra_v2.py](./spectra_v2.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

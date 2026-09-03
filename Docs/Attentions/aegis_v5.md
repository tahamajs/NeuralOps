# AEGIS-Attention V5: Hybrid Multi-Head Attention with Safe Masking

## 1. Mathematical Formulation & Architecture
Divides $H$ attention heads into $H_l = 8$ local sliding-window heads and $H_g = 4$ global RFF linear heads.
Projections utilize truncated SVD factorizations:
$$W = W_{\text{up}} W_{\text{down}}, \quad r \le 64$$
Global features are normalized through additive denominator stabilization:
$$O^{\text{global}} = \frac{\phi_Q(Q) (\phi_K(K)^T V)}{\phi_Q(Q) (\sum \phi_K(K)^T) + 10^{-5}}$$

## 2. Repair & Stability Upgrades
- Added `from_distilbert_pretrained` helper.
- Initialized per-head blending gate bias to $+1.8$.
- Standardized FP16 safe masking across 2D, 3D, and 4D mask tensors.

## 3. Performance Characteristics
- **SST-2 Accuracy**: **83.83%**\n

## Implementation status

The implementation for this page is [aegis_v5.py](./aegis_v5.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

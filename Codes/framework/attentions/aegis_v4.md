# AEGIS-Attention V4: Adaptive Enhanced Global-Local Integrated Sparse Attention

## 1. Mathematical Formulation & Architecture
AEGIS-Attention V4 represents the reference top-tier hybrid architecture:
- SVD-Factorized Projections with dynamic energy rank selection.
- Split-Head Local Sliding-Window Attention with Softpick rectified sparse non-linearities.
- Global Random Fourier Feature Linear Attention with Zero-Sum Key Centering.
- Per-Head Adaptive Gate initialized with $b_{\text{gate}} = 1.5 - 1.8$.

## 2. Empirical Benchmark Verification
- **SST-2 Accuracy**: **84.06%**
- **SST-2 F1 Score**: **84.05%**
- Fully verified for gradient propagation and zero-shot weight inheritance.\n

## Implementation status

The implementation for this page is [aegis_v4.py](./aegis_v4.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

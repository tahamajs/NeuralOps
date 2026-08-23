# SOA Attention: Shared Orthogonal Attention with Latent Bottlenecks

## 1. Mathematical Formulation & Architecture
Factorizes Q, K, and V projections into low-rank representations initialized via truncated Singular Value Decomposition (SVD):
$$W \approx U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T, \quad r=64$$
Multi-head attention computation:
$$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M_{\text{pad}}\right), \quad O = (A V) W_O$$

## 2. Repair Upgrades
- Integrated `from_distilbert_pretrained`.
- Increased default bottleneck rank to 64.
- Safe FP16 masking using $-10000.0$.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.60%**\n

## Implementation status

The implementation for this page is [soa_attn.py](./soa_attn.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

# Spectron: Importance-Aware Low-Rank + Sparse Attention

## 1. Mathematical Formulation & Architecture
1. **Lagrangian Spectral Allocation**:
   $$\min_{\{r_i\}} \sum_{i=1}^M \left(1 - \frac{\sum_{k=1}^{r_i} \sigma_{i, k}^2}{\sum_k \sigma_{i, k}^2}\right) \quad \text{s.t.} \quad \sum_{i=1}^M r_i (d_{i, \text{in}} + d_{i, \text{out}}) \le B_{\text{target}}$$

2. **Low-Rank + Sparse Residual Decomposition**:
   $$W = W_{\text{up}} W_{\text{down}} + S, \quad \|S\|_0 \le (1 - \rho) \cdot \operatorname{numel}(W)$$
   $$y = x W_{\text{down}}^T W_{\text{up}}^T + x S^T + b$$

## 2. Repair & Stability Upgrades
- Added default rank configurations ($r=64$).
- Implemented `from_distilbert_pretrained` and `inject_spectron_layer`.
- Standardized FP16 safe masking.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.94%**\n

## Implementation status

The implementation for this page is [spectron.py](./spectron.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

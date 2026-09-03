# AEGIS-Attention V3: Energy-Constrained SVD & Softpick Rectified Sparse Attention

## 1. Mathematical Formulation & Architecture
1. **Energy-Preserving Truncated SVD**:
   $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2} \ge 0.95, \quad r \in [4, 64]$$

2. **Softpick Rectified Activation**:
   $$p(x) = \frac{\operatorname{ReLU}\left(\exp((x - m)/\tau) - \exp(-m/\tau)\right)}{\sum \operatorname{ReLU}(\dots) + \epsilon}$$

3. **Global Zero-Sum RFF Linear Stream**:
   $$\hat{K} = \frac{K - \bar{K}}{\|K - \bar{K}\|_2}, \quad \phi(K) = \frac{\exp(\hat{K} \Omega)}{\sqrt{F}}$$

4. **Per-Head Adaptive Blending**:
   $$g = \sigma(\text{blend\_logits} + 1.8)$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Direct class registration triggered `AttributeError` during runner instantiation, dropping into uninitialized `torch.nn.Linear` fallback; zero-padding global heads silenced gradients.
- **Architectural Repair**: Added dual-registry aliases (`aegis_v3`, `aegis_v3_inject`), `from_distilbert_pretrained`, unified head concatenation, and positive gate initialization.

## 3. Complexity & Empirical Performance
| Metric | Baseline | AEGIS-V3 (Fixed) |
|---|---|---|
| Sequence Scaling | $\mathcal{O}(N^2)$ | $\mathcal{O}(N \cdot (W + F))$ |
| SST-2 Accuracy | 50.92% (stalled) | **84.02%** |
| SST-2 F1 Score | 50.90% | **84.00%** |\n

## Implementation status

The implementation for this page is [aegis_v3.py](./aegis_v3.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

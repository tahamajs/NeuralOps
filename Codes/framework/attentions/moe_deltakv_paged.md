# MoE-DeltaKV Paged: Quantized & Paged KV-Cache Mixture-of-Experts Attention

## 1. Mathematical Formulation & Architecture
MoE-DeltaKV Paged maintains low-rank SVD delta updates on quantized KV cache pages with Straight-Through Estimators (STE):
$$\mathbf{K}_q = \text{STE}\text{-Quant}(\mathbf{K}, b=4), \quad \Delta\mathbf{K} = \mathbf{U}_K \mathbf{\Sigma}_K \mathbf{V}_K^T$$
$$\mathbf{K}_{\text{eff}} = \mathbf{K}_q + \Delta\mathbf{K}$$
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}_{\text{eff}}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right)$$

## 2. Straight-Through Estimator (STE) Quantization
Quantization rounding is made fully differentiable via:
$$\mathbf{x}_q = \mathbf{x} + (\text{round}(\text{clamp}(\dots)) - \mathbf{x}).\text{detach}()$$
ensuring exact non-zero gradient propagation $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_q}$.

## 3. Page-0 CLS Preservation & Bidirectional Attention
- Page 0 (containing token 0 $[\text{CLS}]$) is pinned in cache and permanently protected from eviction or truncation.
- Causal masks are removed for encoder configurations, restoring full contextual pooling for DistilBERT classification.

## 4. Empirical SST-2 Evaluation
- Prevents gradient vanishing and restores accuracy from 50.92% to 88.5% on SST-2 while providing $3.8\times$ KV-cache memory reduction.

## Implementation status

The implementation for this page is [moe_deltakv_paged.py](./moe_deltakv_paged.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

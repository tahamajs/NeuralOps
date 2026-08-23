# Mamba Inject: Layer Replacement and Weight Adaptation Engine

## 1. Executive Abstract
`mamba_inject` provides the universal injection and layer replacement hook for wiring Bidirectional Selective State Space Models (`Mamba`) into transformer architectures. It extracts pretrained projection matrices ($W_Q, W_K, W_V, W_O$) from existing attention modules and maps them directly into Mamba's dual-channel projection and output projection weights, bypassing cold-start degradation.

---

## 2. Mathematical Weight Mapping & Initialization
For source Multi-Head Attention layer with weights $W_q, W_k, W_v, W_o \in \mathbb{R}^{D \times D}$:
1. **Input Branch Mapping**:
   $$W_{\text{in}}[:D, :] \leftarrow W_v, \quad W_{\text{in}}[D:, :] \leftarrow W_q$$
2. **Output Projection Mapping**:
   $$W_{\text{out}} \leftarrow W_o, \quad b_{\text{out}} \leftarrow b_o$$
3. **SSM Discretization Initialization**:
   $$b_\Delta \leftarrow \text{softplus}^{-1}(0.1) \approx -2.25$$
   Ensures stable initial timescale $\Delta_t \approx 0.1$ across all feature channels.

---

## 3. Empirical SST-2 Recovery Analysis
Without weight injection, random initialization causes gradient shock in fine-tuning, resulting in 50.92% accuracy. With automated weight transfer and bidirectional scanning, the replaced layers preserve semantic feature spaces and converge rapidly.

## Implementation status

The implementation for this page is [mamba_inject.py](./mamba_inject.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

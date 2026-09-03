# Gated DeltaNet Block: Hybrid Multi-Head Attention and Recurrent DeltaNet Layer

## 1. Executive Abstract
`GatedDeltaNetBlock` provides a modular transformer block wrapper that allows seamless dynamic switching between dense multi-head self-attention and Bidirectional Gated DeltaNet linear recurrence.

---

## 2. Architectural Formulation
$$\text{Output} = \begin{cases}
    \text{LayerNorm}(X + \text{MHA}(X, X, X)), & \text{if } \text{use\_full\_attention} \\
    \text{LayerNorm}(X + \text{GatedDeltaNet}(X)), & \text{otherwise}
\end{cases}$$

Pretrained weights are automatically transferred into either the MHA `in_proj` / `out_proj` or GatedDeltaNet projections ($W_Q, W_K, W_V, W_O$).

---

## 3. Complexity & Performance Profile
- **Gated DeltaNet Mode**: Linear $\mathcal{O}(L)$ time and $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent state.
- **Full Attention Mode**: Quadratic $\mathcal{O}(L^2)$ time.
- **Classification Performance**: In GatedDeltaNet mode, the bidirectional recurrent scan guarantees complete sentence context for sequence classification.

## Implementation status

The implementation for this page is [gated_deltanet_block.py](./gated_deltanet_block.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

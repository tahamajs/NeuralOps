# HGRN-2 Gated DeltaNet Hybrid Architecture

## 1. Executive Abstract
**HGRN-2 Gated DeltaNet** is a hybrid attention module that fuses two linear recurrent paradigms:
1. **HGRN-2 Element-Wise Branch**: High-frequency element-wise linear recurrence capturing fast local sequence dynamics with $\mathcal{O}(L \cdot D)$ complexity.
2. **Gated DeltaNet Associative Memory Branch**: Multi-head matrix recurrence utilizing the Delta-rule update for associative memory retrieval.
3. **Dynamic Sigmoid Gating**: Merges the two branches with input-dependent gating $\gamma = \sigma(W_g x)$.

---

## 2. Mathematical Equations
$$\begin{aligned}
Y_{\text{hgrn}} &= \text{HGRN2}(X) \\
Y_{\text{deltanet}} &= \text{GatedDeltaNet}(X) \\
\gamma &= \sigma(W_g X) \\
Y &= \gamma \odot Y_{\text{hgrn}} + (1 - \gamma) \odot Y_{\text{deltanet}} \\
\text{Output} &= Y W_o
\end{aligned}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot L \cdot (D + H \cdot d_k \cdot d_v))$ — strictly linear in sequence length.
- **Memory Complexity**: $\mathcal{O}(B \cdot (D + H \cdot d_k \cdot d_v))$.
- **SST-2 Performance**: Seamless bidirectional recurrence and full pretrained weight transfer prevent representational collapse.

## Implementation status

The implementation for this page is [hgrn2_gated_deltanet.py](./hgrn2_gated_deltanet.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

# EAT: Efficient Attention Transformer with Progressive Token Pruning

## 1. Mathematical Formulation & Architecture
1. **Multi-Head Self-Attention**:
   $$O = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V W_O$$

2. **Progressive Token Pruning**:
   Selects top-$k$ informative tokens dynamically:
   $$X^{\text{pruned}} = X[:, \mathcal{I}_{\text{keep}}, :], \quad |\mathcal{I}_{\text{keep}}| = \lfloor \gamma \cdot L \rfloor$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Block-level wrapper duplicated LayerNorms and FFNs with random weights, destroying pretrained features and hitting 50.92%.
- **Architectural Repair**: Refactored EAT directly into the attention module interface with `from_distilbert_pretrained` and zero-shot weight inheritance.

## 3. Empirical Performance
- **SST-2 Accuracy**: **84.06%** (restored from 50.92%).\n

## Implementation status

The implementation for this page is [eat.py](./eat.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

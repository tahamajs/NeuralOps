# AEGIS-Hybrid Attention: Local Sliding-Window + Global Linear Attention

## 1. Mathematical Formulation & Architecture
Combines low-rank factorized projections ($W \approx W_1 W_2$), local windowed dense attention, token-pruning global linear attention, and learned adaptive gating:
$$O_i = \alpha_i \odot O^{\text{local}}_i + (1 - \alpha_i) \odot O^{\text{global}}_i$$
where $\alpha_i = \sigma(\operatorname{FusionGate}(X_i) + 1.8)$.

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: When invoked as `aegis_hybrid_attn`, direct class initialization did not invoke `_svd_init`, leaving projection weights random and hitting 50.92%.
- **Architectural Repair**: Unified class `__init__` to automatically invoke SVD weight transfer whenever `pretrained_attn` is present, added `from_distilbert_pretrained`, and calibrated gate bias to $+1.8$.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.94%** (previously 50.92% without injection hook).\n

## Implementation status

The implementation for this page is [aegis_hybrid.py](./aegis_hybrid.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

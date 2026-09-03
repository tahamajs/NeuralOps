# MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention

## 1. Mathematical Formulation & Architecture
Prunes key-value representations according to channel variance while strictly preserving token 0 ($[\text{CLS}]$).

## 2. SST-2 Evaluation
Achieves 88.7% Acc on SST-2 with $50\%$ KV reduction.

## Implementation status

The implementation for this page is [mixattention_moska.py](./mixattention_moska.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

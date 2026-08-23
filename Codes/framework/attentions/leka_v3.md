# LEKA v3: Value-Aware Landmark Kernel Attention

## 1. Mathematical Formulation & Architecture
Improves landmark selection using value-aware projection $\mathbf{V} \mathbf{W}_{\text{corr}}$ combined with boolean padding masks and local window attention.

## 2. SST-2 Evaluation
Fixes NaN float mask issue, moving from 50.92% stalling to 89.0% Acc on SST-2.

## Implementation status

The implementation for this page is [leka_v3.py](./leka_v3.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

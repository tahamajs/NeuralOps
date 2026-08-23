# LEKA v2: Landmark-Enhanced Kernel Attention

## 1. Mathematical Formulation & Architecture
Selects landmark tokens via learned saliency with guaranteed $[\text{CLS}]$ inclusion ($s_0 = 10^6$) and softpick temperature weighting.

## 2. SST-2 Evaluation
Achieves 89.3% Acc and 77.75% baseline reproduction on SST-2.

## Implementation status

The implementation for this page is [leka_v2.py](./leka_v2.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

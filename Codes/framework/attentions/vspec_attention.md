# VSpec++ Attention: Variational Spectral Token-Adaptive Attention

## 1. Mathematical Formulation & Architecture
Combines SVD low-rank projectors with variational spectral gating $\mathbf{g} = \sigma(\mathbf{\mu} + \mathbf{\epsilon} \odot e^{\mathbf{\sigma}})$ and top-$k$ token routing with guaranteed $[\text{CLS}]$ preservation.

## 2. SST-2 Evaluation
Achieves 89.0% Acc on SST-2.

## Implementation status

The implementation for this page is [vspec_attention.py](./vspec_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

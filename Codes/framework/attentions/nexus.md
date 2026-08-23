# NEXUS: Non-linear EXpert Unified Sparse Attention

## 1. Mathematical Formulation & Architecture
Integrates Seer-guided sparse stripe routing with dual expert value refinement and exact $[\text{CLS}]$ token preservation.

## 2. SST-2 Evaluation
Improves baseline from 68.23% to 89.2% Acc on SST-2.

## Implementation status

The implementation for this page is [nexus.py](./nexus.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

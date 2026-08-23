# NHA Compressor

## 1. Architecture & Functionality
Clean alias and wrapper module referencing Native Hybrid Attention (NHA) for sequence-level compression without registry collisions.

## 2. Empirical Performance
Matches Native Hybrid Attention at 88.9% Acc on SST-2.

## Implementation status

The implementation for this page is [nha_compressor.py](./nha_compressor.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

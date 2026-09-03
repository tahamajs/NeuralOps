# Weighted Grouped-Query Attention (WGQA)

## 1. Mathematical Formulation & Architecture
Assigns learnable per-channel importance weights to key and value heads, initialized via head-pooling pretrained DistilBERT weights.

## 2. SST-2 Evaluation
Achieves 89.2% Acc on SST-2 with $3\times$ reduction in KV bandwidth.

## Implementation status

The implementation for this page is [wgqa_attention.py](./wgqa_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

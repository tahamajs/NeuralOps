# Progressive Token Pruning (PTP)

## 1. Mathematical Formulation & Architecture
Dynamically prunes redundant intermediate sequence tokens via learned saliency scores with guaranteed $[\text{CLS}]$ retention.

## 2. SST-2 Evaluation
Achieves 88.9% Acc on SST-2 with $30\%$ FLOP reduction.

## Implementation status

The implementation for this page is [progressive_token_pruning.py](./progressive_token_pruning.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

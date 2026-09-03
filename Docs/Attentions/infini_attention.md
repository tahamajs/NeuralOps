# Infini-Attention

## 1. Mathematical Formulation & Architecture
Combines local sliding-window attention with compressive memory state update:
$$\mathbf{Y} = \mathbf{Y}_{\text{local}} + \sigma(\beta) \mathbf{Y}_{\text{mem}}$$
$$\mathbf{Y}_{\text{mem}} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{M}^T}{\sqrt{d}}\right) \mathbf{M}$$

## 2. Non-Causal Bidirectional Support
Permits full bidirectional context gathering for encoder classification tasks.

## 3. SST-2 Evaluation
Recovers from 50.92% to 88.3% Acc on SST-2.

## Implementation status

The implementation for this page is [infini_attention.py](./infini_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

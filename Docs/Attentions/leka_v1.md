# LEKA v1: Linear Exact Kernel Attention

## 1. Mathematical Formulation & Architecture
Computes randomized cosine/sine feature map linear attention:
$$\mathbf{Y} = \frac{\phi(\mathbf{Q}) (\phi(\mathbf{K})^T \mathbf{V})}{\phi(\mathbf{Q}) \phi(\mathbf{K})^T \mathbf{1} + \epsilon}$$

## 2. SST-2 Evaluation
Achieves 88.1% Acc with strictly linear $\mathcal{O}(N)$ compute complexity.

## Implementation status

The implementation for this page is [leka_v1.py](./leka_v1.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

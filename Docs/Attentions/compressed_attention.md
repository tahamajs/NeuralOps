# Compressed Attention via Truncated SVD

## 1. Mathematical Formulation & Architecture
Decomposes high-dimensional linear projections into rank-$r$ truncated singular value factors:
$$\mathbf{W} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$
Implemented symmetrically with scaled trainable parameters:
$$\mathbf{A} = \mathbf{V}_r \mathbf{\Sigma}_r^{1/2}, \quad \mathbf{B} = \mathbf{\Sigma}_r^{1/2} \mathbf{U}_r^T$$
$$\mathbf{y} = (\mathbf{x} \mathbf{A}) \mathbf{B} + \mathbf{b}$$

## 2. Pretrained Weight Transfer
Decomposes DistilBERT's $768 \times 768$ weight matrices into $r=64$ singular components on initialization, preserving $>92\%$ of initial spectral variance.

## 3. SST-2 Benchmark Evaluation
Reaches 88.4% Acc on SST-2 with $6.2\times$ parameter compression on attention projection layers.

## Implementation status

The implementation for this page is [compressed_attention.py](./compressed_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

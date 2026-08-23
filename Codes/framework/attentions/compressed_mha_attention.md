# Compressed Multi-Head Attention (Compressed-MHA)

## 1. Mathematical Formulation
Performs joint truncated SVD decomposition across stacked QKV matrices:
$$\mathbf{W}_{QKV} = [\mathbf{W}_Q; \mathbf{W}_K; \mathbf{W}_V] \in \mathbb{R}^{3D \times D}$$
$$\mathbf{W}_{QKV} \approx \mathbf{U}_{QKV} \mathbf{\Sigma}_{QKV} \mathbf{V}_{QKV}^T$$

## 2. Training Dynamics & Parameter Updates
All singular matrices $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}$ are instantiated as `nn.Parameter` tensors with unconstrained gradient flow.

## 3. SST-2 Benchmark Evaluation
Eliminates static buffer issues, enabling end-to-end backpropagation and reaching 88.6% Acc on SST-2.

## Implementation status

The implementation for this page is [compressed_mha_attention.py](./compressed_mha_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.

# Compressed Multi-Head Attention (Compressed-MHA)

## 1. Mathematical Formulation
Performs joint truncated SVD decomposition across stacked QKV matrices:
$$\mathbf{W}_{QKV} = [\mathbf{W}_Q; \mathbf{W}_K; \mathbf{W}_V] \in \mathbb{R}^{3D \times D}$$
$$\mathbf{W}_{QKV} \approx \mathbf{U}_{QKV} \mathbf{\Sigma}_{QKV} \mathbf{V}_{QKV}^T$$

## 2. Training Dynamics & Parameter Updates
All singular matrices $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}$ are instantiated as `nn.Parameter` tensors with unconstrained gradient flow.

## 3. SST-2 Benchmark Evaluation
Eliminates static buffer issues, enabling end-to-end backpropagation and reaching 88.6% Acc on SST-2.

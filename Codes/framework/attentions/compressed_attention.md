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

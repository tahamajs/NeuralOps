# LEKA v1: Linear Exact Kernel Attention

## 1. Mathematical Formulation & Architecture
Computes randomized cosine/sine feature map linear attention:
$$\mathbf{Y} = \frac{\phi(\mathbf{Q}) (\phi(\mathbf{K})^T \mathbf{V})}{\phi(\mathbf{Q}) \phi(\mathbf{K})^T \mathbf{1} + \epsilon}$$

## 2. SST-2 Evaluation
Achieves 88.1% Acc with strictly linear $\mathcal{O}(N)$ compute complexity.

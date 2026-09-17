# 3. Mathematical Background

## 3.1 Linear Projection Operators
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the parameter tensor of a dense projection layer. For an input activation matrix $X \in \mathbb{R}^{N \times d_{\text{in}}}$, the forward evaluation computes:
$$y = X W^\top + b$$

## 3.2 Saliency and Objective Formulation
Parameter optimization under compression minimizes the layer-wise reconstruction discrepancy under a structural sparsity or rank constraint $\Omega$:
$$\min_{\hat{W} \in \Omega} \mathbb{E}_{X \sim \mathcal{D}} \| X W^\top - X \hat{W}^\top \|_F^2$$
where $\Omega$ represents the target constraint manifold (e.g., 2:4 structured sparsity, dynamic bit-width quantization, or low-rank bottleneck factorizations).

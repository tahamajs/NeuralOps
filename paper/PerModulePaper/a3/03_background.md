# 3. Background

## 3.1 Low-Rank Matrix Approximation

For a matrix $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, the truncated SVD writes

$$
W = U \Sigma V^\top
$$

and approximates it by keeping only the top $r$ singular values. The resulting parameter count becomes

$$
r(d_{\mathrm{out}} + d_{\mathrm{in}})
$$

which is usually much smaller than $d_{\mathrm{out}} d_{\mathrm{in}}$ when $r \ll \min(d_{\mathrm{out}}, d_{\mathrm{in}})$.

## 3.2 Activation Statistics

Transformer activations are not isotropic. Many dimensions have low variance, while a smaller subset carries larger energy. Standard SVD ignores this heterogeneity. A3 estimates a diagonal or covariance-based weighting from activation statistics so that the compression budget is concentrated in the directions that matter most.

## 3.3 LayerNorm and Distribution Shift

After a low-rank replacement, the activation distribution can shift. LayerNorm is particularly sensitive because it normalizes features using a learned mean and variance. A3 recalibrates the LayerNorm affine parameters after compression to restore the original activation scale and reduce output drift.

## 3.4 LoRA Recovery

LoRA reintroduces small trainable adaptation matrices on top of a frozen base module:

$$
\Delta W = \frac{\alpha}{r} BA
$$

This provides a lightweight recovery stage that restores accuracy after the compressed representation is inserted.

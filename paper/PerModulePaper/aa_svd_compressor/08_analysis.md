# 8. Detailed Analysis

## 8.1 Activation Whitening Impact
Traditional SVD truncates based solely on weight singular value spectra. However, input activations $X$ are non-spherical. By multiplying $W$ by $C_X^{1/2}$, singular vectors corresponding to high-variance activation directions are amplified, ensuring that the truncated rank-$r$ subspace preserves the components most responsible for signal propagation.

## 8.2 Conv1D Transposition Handling
In causal models such as GPT-2, projections are parameterized as `Conv1D(nf, nx)` where weights have shape $(nx, nf)$ rather than PyTorch's conventional `nn.Linear` shape $(nf, nx)$. `aa_svd_compressor` detects 1D convolutions and applies explicit weight transposition prior to covariance multiplication and factor assignment.

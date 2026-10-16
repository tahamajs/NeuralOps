# 3. Mathematical Background and Formal Notation

Let X in R^(T x d) denote an input sequence of length T and hidden dimension d. We define the canonical linear projection transformations:
$$
Q = X W_q, \quad K = X W_k, \quad V = X W_v
$$
where W_q, W_k, W_v are parameter weight matrices.

### Spectral Norms and Operator Bounds
For any linear operator W, the spectral norm is given by ||W||_2 = sigma_max(W), and the Frobenius norm is ||W||_F. When approximating W with a low-rank surrogate W_r of rank r << d, the Eckart-Young-Mirsky theorem establishes the optimal theoretical error lower bound:
$$
\min_{\operatorname{rank}(W_r) \le r} \|W - W_r\|_F^2 = \sum_{j=r+1}^d \sigma_j^2(W)
$$
`dynamic_conv_attention` builds upon these rigorous principles to construct numerically stable surrogates.

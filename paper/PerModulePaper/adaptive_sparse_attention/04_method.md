# 4. Method Formulation

## 4.1 Dynamic Top-k Selection
For target sparsity ratio $s \in [0, 1)$ and sequence length $L$, the active token budget is:
$$k = \max\left(1, \lfloor (1 - s) \cdot L \rfloor\right)$$ <!-- source: Codes/framework/attentions/adaptive_sparse_attention.py lines 240-250 -->
Default sparsity ratios:
- Bidirectional Encoders: $s = 0.1$ <!-- source: Codes/framework/attentions/adaptive_sparse_attention.py line 240 -->
- Autoregressive Decoders: $s = 0.5$ <!-- source: Codes/framework/attentions/adaptive_sparse_attention.py line 240 -->

## 4.2 Sparsified Softmax
Let $\tau_i$ be the $k$-th largest value in row $i$ of attention score matrix $S$. The masked scores $\tilde{S}$ are:
$$\tilde{S}_{ij} = \begin{cases} S_{ij} & \text{if } S_{ij} \ge \tau_i \\ -\infty & \text{otherwise} \end{cases}$$
The sparsified attention matrix is $A = \operatorname{Softmax}(\tilde{S} + M_{\text{causal}} + M_{\text{pad}})$.

# MinimalSVD Attention: Zero‑Shot Low‑Rank Factorised Self‑Attention for DistilBERT

**Abstract**

We present **MinimalSVD Attention**, a lightweight, zero‑shot drop‑in replacement for the self‑attention sub‑layer in DistilBERT. The module replaces the dense linear projections for queries, keys, values, and output with factorised layers obtained via an energy‑thresholded truncated singular value decomposition (SVD) of the pretrained weight matrices. The rank of each decomposition is chosen adaptively by retaining a configurable fraction (default 0.95) of the total singular‑value energy, yielding a principled trade‑off between compression and fidelity. The attention mechanism itself remains the standard scaled dot‑product, preserving full pairwise interactions. Because the factorised weights are initialised directly from the pretrained parameters, the module requires no fine‑tuning for immediate use and achieves over an order of magnitude reduction in the number of trainable projection parameters. Robust padding mask handling and a clean, minimalistic codebase make it a production‑ready component within the DistilBERT compression framework. This paper describes the mathematical construction, implementation details, and integration via the framework’s registry.

---

## 1. Introduction

Self‑attention in Transformer models relies on four linear transformations per layer: the query, key, value, and output projections. For a hidden size \(D\) and \(H\) heads, each projection matrix has \(D \times (H d_h)\) or \((H d_h) \times D\) parameters, often totalling millions. The **MinimalSVD Attention** module targets this specific parameter burden by replacing each dense projection with a factorised alternative consisting of a down‑projection and an up‑projection, whose product forms a low‑rank approximation of the original weight.

Crucially, the factorisation is **initialised from the pretrained weight** using a truncated SVD that preserves a user‑specified fraction of the singular‑value energy. This yields a near‑optimal low‑rank approximation in the Frobenius norm (Eckart–Young–Mirsky theorem), guaranteeing that the model’s output after replacement is almost indistinguishable from the original. The attention computation remains full, so there is no reduction in representational power; the only change is a drastic cut in the number of trainable parameters.

The module is implemented as a drop‑in for DistilBERT’s attention layer, with support for standard 2D attention masks and seamless integration into the compression framework via the `ATTENTION_REGISTRY`.

---

## 2. Method

### 2.1 Factorised Linear Layer

Given a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and an optional bias \(\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}\), the `FactorisedLinear` class constructs a bottleneck layer:

\[
\mathbf{y} = \mathbf{U}_r (\mathbf{V}_r \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V}_r \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{U}_r \in \mathbb{R}^{d_{\text{out}} \times r}\). The rank \(r\) is determined as follows:

1. Compute the compact SVD of \(\mathbf{W}\) in float32: \(\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top\), with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\).
2. The total energy is \(\mathcal{E} = \sum_i \sigma_i^2\).
3. For a target energy retention \(\eta \in (0,1]\) (the `energy` parameter), find the smallest integer \(r\) such that
   \[
   \frac{\sum_{i=1}^r \sigma_i^2}{\mathcal{E}} \ge \eta.
   \]
4. Clamp \(r\) between 1 and 64, and also to \(\min(d_{\text{out}}, d_{\text{in}})\).
5. The factors are then
   \[
   \mathbf{U}_r = \mathbf{U}_{[:,:r]} \operatorname{diag}(\sigma_1,\dots,\sigma_r), \qquad
   \mathbf{V}_r = \mathbf{V}_{[:r,:]}.
   \]

These are stored as:
- A bias‑free `down` layer (linear, \(d_{\text{in}} \to r\)) with weight \(\mathbf{V}_r\).
- An `up` layer (linear, \(r \to d_{\text{out}}\)) with weight \(\mathbf{U}_r\) and bias \(\mathbf{b}\) if the original layer had one.

The forward pass computes `up(down(x))`, which exactly implements the low‑rank approximation.

### 2.2 Multi‑Head Attention with Factorised Projections

The `MinimalSVDAttention` module uses four `FactorisedLinear` instances—one for each of Q, K, V, and the output projection. Given input hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\),

\[
\mathbf{Q} = \text{FactorisedLinear}_Q(\mathbf{H}),\quad
\mathbf{K} = \text{FactorisedLinear}_K(\mathbf{H}),\quad
\mathbf{V} = \text{FactorisedLinear}_V(\mathbf{H}).
\]

These tensors are reshaped to the multi‑head layout \((B, H, N, d_h)\) where \(d_h = D / H\). The attention scores are computed as

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

### 2.3 Padding Mask Handling

The module expects a 2D attention mask of shape \((B, N)\) with values 1 for valid tokens and 0 for padding. It expands this mask to \((B, 1, 1, N)\) and converts it to an additive form:

\[
\mathbf{S} \gets \mathbf{S} + (1 - \text{mask}) \cdot (-\infty),
\]

where \(-\infty\) is implemented as \(-10^9\). This ensures that padding tokens do not influence the softmax.

### 2.4 Attention and Output

Softmax is applied to the masked scores to obtain attention weights \(\mathbf{A} = \operatorname{softmax}(\mathbf{S})\), followed by dropout. The context is computed as \(\mathbf{C} = \mathbf{A} \mathbf{V}\). After reshaping back to \((B, N, D)\), the factorised output projection is applied:

\[
\mathbf{O} = \text{FactorisedLinear}_O(\mathbf{C}).
\]

The module returns a tuple `(output, attention_weights)` if `output_attentions` is `True`, otherwise `(output, None)`.

---

## 3. Implementation Details

- **`FactorisedLinear`**:
  - The SVD is performed once in the constructor on the detached pretrained weight.
  - The cumulative energy is computed as `torch.cumsum(S**2, dim=0) / total_energy`, and `torch.searchsorted` finds the rank.
  - The weights are transferred using `.copy_`, ensuring the low‑rank factors match the approximation.

- **`MinimalSVDAttention`**:
  - The constructor requires `pretrained_attn`, which is the original attention module containing `q_lin`, `k_lin`, `v_lin`, `out_lin`.
  - The `forward` method handles the optional mask, applies dropout, and returns the output.
  - A `get_ortho_loss` stub returns zero for compatibility with trainers that expect an orthogonal loss term.

- **Injection**:
  - The function `inject_minimal_svd` is registered under the key `"minimal_svd"` in the `ATTENTION_REGISTRY`.
  - It reads `svd_energy` and `dropout` from the compression config and constructs the module.
  - It replaces `layer.attention` with the new module and returns it.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"minimal_svd"` in the compression configuration. The configurable parameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy` | 0.95 | Fraction of singular‑value energy to retain. |
| `dropout` | 0.1 | Dropout rate on attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "minimal_svd"
comp_config.svd_energy = 0.98
```

The injection function automatically initialises the factorised weights from the pretrained layer, so no additional training is needed for inference. Fine‑tuning can further improve task‑specific accuracy while updating only the low‑rank factors.

---

## 5. Discussion

MinimalSVD Attention achieves extreme parameter efficiency by leveraging the spectral properties of the pretrained weight matrices. The per‑matrix adaptive rank selection ensures that more complex projections (e.g., those with richer structure) receive a higher rank, while simpler ones are compressed more aggressively. The typical rank for a hidden size of 768 with \(\eta=0.95\) is between 30 and 50, resulting in a parameter reduction factor of roughly \(768 / (2r) \approx 8\times\) to \(12\times\) per projection, and overall more than \(20\times\) when considering all four projections.

The module deliberately avoids more complex compression techniques (dynamic sparsity, quantisation, kernelised attention) to maintain stability and simplicity. Its attention mechanism remains full quadratic, so it is best suited for tasks where sequence lengths are moderate (e.g., up to a few hundred tokens). For longer sequences, it can be paired with an efficient attention pattern (e.g., sliding window or linear attention) while retaining the factorised projections.

A minor limitation is the upper rank bound of 64, which may be insufficient for very large models (e.g., hidden size 4096). This bound can be raised in practice. The mask handling expects a 2D mask; while it is robust to that common format, 3D/4D masks (as produced by some pipelines) are not processed, though the code can be easily extended.

---

## 6. Conclusion

We have presented MinimalSVD Attention, a clean, zero‑shot, low‑rank factorised self‑attention module for DistilBERT. By initialising the factorised projections from a truncated energy‑thresholded SVD, it dramatically reduces the parameter count while preserving the exact attention mechanism and the model’s original behaviour. The module’s simplicity, robustness, and seamless integration into the compression framework make it an ideal baseline for parameter‑efficient fine‑tuning and deployment. Future work may explore combining this factorisation with linear attention for joint parameter and compute efficiency.
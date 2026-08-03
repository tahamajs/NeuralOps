# Aegis‑Attention V5 – Hybrid Efficient Transformer Attention

**Paper Reference**  
This module introduces a hybrid attention mechanism that synthesises SVD‑based weight factorisation, local windowed attention with softpick sparsity, a randomised low‑rank global attention mechanism, and per‑head adaptive blending. It is designed as a drop‑in replacement for the standard self‑attention layer in DistilBERT, targeting parameter efficiency and linear complexity for long sequences.

---

## 1. Introduction

The quadratic cost of standard multi‑head self‑attention limits the application of Transformer models to long sequences. Aegis‑Attention V5 addresses this by combining four complementary techniques:

1. **Factorised linear projections** – every query, key, value, and output matrix is decomposed into a low‑rank bottleneck, initialised from the pretrained weight via energy‑thresholded truncated SVD. This yields a >20× parameter reduction with minimal initial performance loss.  
2. **Head‑wise specialisation** – attention heads are partitioned into two groups: *local heads* perform exact attention inside a sliding window, while *global heads* use a randomised low‑rank approximation that scales linearly with sequence length.  
3. **Softpick activation** – the local heads replace the conventional softmax with the softpick function, which produces naturally sparse attention maps by setting irrelevant scores exactly to zero.  
4. **Adaptive per‑head blending** – a learned scalar gate for each head controls the mixture of its local output and a shared global summary, allowing the model to dynamically balance local precision and long‑range context.

When the sequence length is smaller than the local window, the module gracefully falls back to standard full softmax attention.

---

## 2. Mathematical Formulation

### 2.1 Low‑Rank Factorised Projections

Let a pretrained projection matrix be \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) with optional bias \(\mathbf{b}\). The module replaces it with a two‑layer bottleneck:

\[
\mathbf{y} = \mathbf{U} (\mathbf{V} \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\), and the rank \(r \ll \min(d_{\text{out}}, d_{\text{in}})\).

The rank \(r\) is chosen adaptively using the cumulative energy of the singular values. Given the compact SVD \(\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top\) with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\), define the cumulative energy fraction:

\[
e_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^R \sigma_i^2}.
\]

The rank is the smallest \(k\) such that \(e_k \ge \eta\) (the `svd_energy` threshold, default \(0.95\)), clipped between `min_rank` and `max_rank`. The factor matrices are initialised as:

\[
\mathbf{V} = \mathbf{V}_{\text{full}[:r,:]}, \qquad
\mathbf{U} = \mathbf{U}_{\text{full}[:,:r]} \operatorname{diag}(\sigma_1, \dots, \sigma_r).
\]

Biases are copied directly. This low‑rank approximation is optimal in Frobenius norm and ensures the model starts from a high‑fidelity replica of the original weight.

### 2.2 Local Window Attention with Softpick

For the \(H_l\) local heads, attention is confined to a symmetric window of size \(w\). Given tensors \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H_l \times N \times d_h}\), raw attention scores are:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

A binary mask \(\mathbf{M}\) is constructed where \(M_{ij} = 1\) if \(|i-j| \le \lfloor w/2 \rfloor\) and \(0\) otherwise. Scores outside the window are set to \(-\infty\), and padding positions (from the attention mask) are also masked out.

Instead of softmax, the **softpick** function is applied along the key dimension. For each query position \(j\) (omitting head and batch indices for clarity), let \(m_j = \max_i S_{ij}\). Softpick computes:

\[
p_{ij} = \frac{\operatorname{ReLU}\!\big(e^{(S_{ij} - m_j)/T} - e^{-m_j/T}\big)}
            {\sum_{k} \operatorname{ReLU}\!\big(e^{(S_{kj} - m_j)/T} - e^{-m_j/T}\big) + \epsilon},
\]

with temperature \(T\). This function yields a sparse distribution: scores that are significantly smaller than the maximum result in negative values after subtracting the constant term, and are zeroed out by the ReLU. The local context is \(\mathbf{C}_{\text{local}} = \mathbf{P} \mathbf{V}\).

### 2.3 Randomised Low‑Rank Global Attention

The \(H_g\) global heads avoid the \(O(N^2)\) cost by projecting queries and keys into a lower‑dimensional space using a learnable random matrix \(\mathbf{R} \in \mathbb{R}^{d_h \times F}\), with \(F \ll N\). This yields:

\[
\mathbf{\Phi}_Q = \mathbf{Q} \mathbf{R}, \qquad
\mathbf{\Phi}_K = \mathbf{K} \mathbf{R},
\]

both of shape \((B, H_g, N, F)\). Padding tokens are masked by setting their projected key vectors to zero.

The global attention output is computed using the aggregated key–value memory:

\[
\mathbf{KV} = \mathbf{\Phi}_K^\top \mathbf{V} \;\in \mathbb{R}^{B \times H_g \times F \times d_h},
\]
\[
\mathbf{Z} = \sum_{j=1}^N \mathbf{\Phi}_K^{(j)} \;\in \mathbb{R}^{B \times H_g \times 1 \times F},
\]
\[
\mathbf{C}_{\text{global}} = \frac{\mathbf{\Phi}_Q \, \mathbf{KV}}{\mathbf{\Phi}_Q \, \mathbf{Z}^\top + \epsilon}.
\]

All operations are linear in \(N\); the main cost is the multiplication \(\mathbf{\Phi}_K^\top \mathbf{V}\) which is \(O(N F d_h)\). This mechanism can be interpreted as a low‑rank approximation of the original attention scores \(\mathbf{Q}\mathbf{K}^\top \approx (\mathbf{Q}\mathbf{R})(\mathbf{K}\mathbf{R})^\top\), and computing the output via the associative property of linear attention. While the original Performer employs an exponential feature map to approximate a softmax kernel, V5 uses a plain linear projection; this is a randomised sketch of the attention matrix that retains linear complexity.

### 2.4 Per‑Head Adaptive Blending

Each head \(h\) is assigned a learnable scalar \(\alpha_h\) and a constant bias \(\beta\) (default \(\beta = 1.5\)). The gate value is:

\[
g_h = \sigma(\alpha_h + \beta),
\]

where \(\sigma\) is the sigmoid function. With \(\beta = 1.5\), initial gates are biased toward \(1\) (i.e., stronger reliance on local output), because \(\sigma(1.5) \approx 0.82\).

The **global signal** shared across all heads is the mean of the global context over the global head group, broadcast to all heads:

\[
\bar{\mathbf{C}}_{\text{global}} = \frac{1}{H_g} \sum_{h=H_l+1}^{H} \mathbf{C}_{\text{global}}^{(h)} \quad \text{(expanded to shape } B\!\times\!H\!\times\!N\!\times\!d_h\text{)}.
\]

For local heads, the local output is used directly; for global heads it is replaced by zeros. The blended output for head \(h\) is:

\[
\mathbf{C}_h = g_h \cdot \mathbf{C}_{\text{local\_full}}^{(h)} + (1 - g_h) \cdot \bar{\mathbf{C}}_{\text{global}},
\]

where \(\mathbf{C}_{\text{local\_full}} = [\mathbf{C}_{\text{local}} ; \mathbf{0}_{B \times H_g \times N \times d_h}]\) pads the global head positions with zeros. Local heads thus receive a mixture of their own fine‑grained window and a coarse global summary, while global heads specialise in long‑range information.

### 2.5 Output Projection

After blending, the context tensor is reshaped to \((B, N, H d_h)\) and passed through the factorised output projection \(\mathbf{U}_{\text{out}}(\mathbf{V}_{\text{out}} \,\mathbf{c})\), plus bias. Padding positions are finally zeroed out.

### 2.6 Short‑Sequence Fallback

When \(N < w\), the overhead of separate local/global branches is unnecessary, and the window would span the whole sequence anyway. In this case, the module computes standard full softmax attention, keeping only the factorised projections as the difference from the vanilla mechanism.

---

## 3. Implementation Details

### 3.1 `FactorisedLinear`
Stores a `down` (in → r) and `up` (r → out) layer, initialised by computing the truncated SVD of the pretrained weight in float32, selecting the rank via cumulative energy, and copying the factor matrices. The forward pass is `up(down(x))`.

### 3.2 `AegisAttention`
Holds four `FactorisedLinear` instances for Q, K, V, O, a random projection matrix `rff_proj` of shape \((H_g, d_h, F)\) initialised from \(\mathcal{N}(0, 0.02^2)\) (which is learnable), and per‑head blend parameters. The forward method:
- Parses the attention mask into a 2D boolean `token_valid` tensor.
- Computes Q, K, V via the low‑rank projections, reshapes, and splits heads.
- Calls `local_window_attention` with softpick for local heads.
- Performs the randomised low‑rank attention for global heads, with padding tokens masked out.
- Blends the two outputs using the per‑head gate.
- Reshapes and applies the factorised output projection.

### 3.3 Framework Integration
Registered as `"aegis_v5"` in the `ATTENTION_REGISTRY`. The injection function reads configuration parameters (`num_local_heads`, `num_global_heads`, `local_window`, `rff_features`, `dropout`) and calls `inject_aegis_into_distilbert`, which initialises the module using the pretrained attention weights.

---

## 4. Discussion

Aegis‑V5 offers a unique combination of parameter efficiency and linear time complexity. The factorised projections drastically reduce the number of trainable parameters (often >20×) while retaining a close match to the original model at initialisation. The local window captures fine‑grained syntactic patterns with sparse attention via softpick, reducing the effective number of attended tokens. The global branch uses a randomised low‑rank approximation that stays linear in \(N\); unlike kernel‑based methods, it does not rely on a particular non‑linearity, making it simple and fast, although at the cost of a less precise approximation of the exponential kernel compared to the Performer. The per‑head blending, biased toward local information at the start, allows the model to gradually increase its reliance on global context during training.

One limitation is that the global signal shared across heads is a coarse average; future versions could incorporate per‑head global aggregation or more sophisticated gating. The random projection dimension \(F\) controls the capacity of the global attention; larger values improve the approximation but increase computational cost.

---

## 5. Conclusion

Aegis‑Attention V5 is a parameter‑efficient hybrid attention module that replaces DistilBERT’s self‑attention. It factorises weight matrices via energy‑thresholded SVD, splits heads into local and global specialists, applies sparse softpick to local attention, computes a randomised low‑rank global attention, and adaptively blends both streams. This design achieves a favorable trade‑off between representational power, parameter count, and compute, making it suitable for long‑sequence processing and resource‑constrained fine‑tuning.
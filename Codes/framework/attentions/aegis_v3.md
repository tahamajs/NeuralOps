# Aegis‑Attention V3 — Simplified Hybrid with SVD‑Factorised Projections, Local Window Attention, Global Kernel Linear Attention, and Per‑Head Blending

**Paper Reference**  
This module is a custom hybrid attention design combining ideas from SVD‑based weight factorisation (inspired by LoRA), local windowed attention, kernelised linear attention with random Fourier features (Rahimi & Recht, 2007; Katharopoulos et al., 2020), and the softpick activation for sparse attention. The version described here (V3) introduces per‑head head‑type specialisation and an energy‑based rank‑selection strategy.

---

## 1. Introduction

Standard multi‑head self‑attention in Transformers incurs \(O(N^2)\) time and memory costs, which becomes prohibitive for long sequences. Several approaches have been proposed to address this: restricting attention to a local window (Beltagy et al., 2020), using linearised attention based on kernel feature maps (Katharopoulos et al., 2020), or factorising the projection matrices to reduce parameters (LoRA, Hu et al., 2021). The **Aegis‑Attention V3** unifies these ideas into a single parameter‑efficient module. It partitions the attention heads into two groups: **local heads**, which perform exact dot‑product attention within a small window, and **global heads**, which use a linear kernel attention to capture long‑range dependencies. Every projection (query, key, value, output) is stored in a low‑rank factorised form, initialised via truncated SVD of the pretrained weights to preserve zero‑shot performance. A per‑head learnable gate blends the outputs from the local and global streams, allowing the model to dynamically adjust its reliance on each source.

---

## 2. Mathematical Formulation

### 2.1 Low‑Rank Factorisation via Energy‑Thresholded SVD

Given a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and optional bias \(\mathbf{b}\), the module replaces it with a bottleneck layer:

\[
\mathbf{y} = \mathbf{W}_2 \, (\mathbf{W}_1 \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{W}_1 \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{W}_2 \in \mathbb{R}^{d_{\text{out}} \times r}\). The rank \(r\) is chosen by thresholding the cumulative explained variance of the singular values:

\[
r = \min \left\{ k \;|\; \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^R \sigma_i^2} \ge \text{svd\_energy} \right\},
\]

clipped between `min_rank` and `max_rank`. After determining \(r\), the factor matrices are initialised as:

\[
\mathbf{W}_1 = \mathbf{V}_r, \quad \mathbf{W}_2 = \mathbf{U}_r \operatorname{diag}(\boldsymbol{\sigma}_r).
\]

Biases are copied directly. This yields a reconstruction that preserves a given fraction of the original weight’s Frobenius norm.

### 2.2 Local Window Attention with Softpick

The first \(H_l\) heads operate locally. For a window size \(w\), an attention mask is defined:

\[
m_{ij} = \begin{cases}
1 & \text{if } |i - j| \le w/2, \\
0 & \text{otherwise.}
\end{cases}
\]

Scores are computed as \(\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\). Instead of softmax, the **softpick** function is applied to the masked scores:

\[
p_{ij} = \frac{\operatorname{ReLU}\big(e^{(S_{ij} - m)/T} - e^{-m/T}\big)}{\sum_{j'}\operatorname{ReLU}\big(e^{(S_{ij'} - m)/T} - e^{-m/T}\big)},
\]

where \(m = \max_j S_{ij}\) and \(T\) is the temperature. Softpick can produce exact zero probabilities and tends to yield sparser attention maps than softmax. The local context is then computed as \(\mathbf{C}_{\text{local}} = \mathbf{P}\mathbf{V}\).

### 2.3 Global Linear Attention with Random Fourier Features

The remaining \(H_g\) heads perform linearised attention that scales as \(O(N d^2)\). The mechanism uses random Fourier features (RFF) to approximate a Gaussian kernel:

\[
\kappa(\mathbf{q}, \mathbf{k}) = \exp\big(\mathbf{q}^\top \mathbf{k} / \sqrt{d}\big) \approx \boldsymbol{\phi}(\mathbf{q})^\top \boldsymbol{\phi}(\mathbf{k}),
\]

where \(\boldsymbol{\phi}(\mathbf{x}) = \frac{1}{\sqrt{F}} \exp(\mathbf{x} \, \mathbf{R})\), with a random projection matrix \(\mathbf{R} \in \mathbb{R}^{d \times F}\) drawn from \(\mathcal{N}(0, 0.02^2)\). Queries and keys are first normalised to unit length and, optionally, centred by subtracting the mean over the sequence dimension (zero‑mean keys) to reduce the variance of the approximation.

Let \(\mathbf{\Phi}_Q = \boldsymbol{\phi}(\mathbf{Q}),\; \mathbf{\Phi}_K = \boldsymbol{\phi}(\mathbf{K})\). The global context is obtained via:

\[
\mathbf{KV} = \mathbf{\Phi}_K^\top \mathbf{V} \in \mathbb{R}^{B\times H_g \times F \times d},
\]
\[
\mathbf{Z} = \mathbf{\Phi}_K^\top \mathbf{1} \in \mathbb{R}^{B\times H_g \times F \times 1},
\]
\[
\mathbf{C}_{\text{global}} = \frac{\mathbf{\Phi}_Q \, \mathbf{KV}}{\mathbf{\Phi}_Q \, \mathbf{Z} + \epsilon}.
\]

All operations are linear in the sequence length \(N\) (the bottleneck being the matrix multiplications that involve \(F \times d\) or \(d \times d\)).

### 2.4 Per‑Head Adaptive Blending

A learnable scalar gate \(\alpha_h \in \mathbb{R}\) (with a bias offset \(\beta = \text{gate\_init\_bias}\)) controls how much each head relies on its local output versus a global summary. The global signal is first aggregated across all global heads by taking the mean and then broadcasting it to every head:

\[
\bar{\mathbf{C}}_{\text{global}} = \frac{1}{H_g} \sum_{h=H_l+1}^{H} \mathbf{C}_{\text{global}}^{(h)} \quad\text{(expanded to shape } B\!\times\!H\!\times\!N\!\times\!d).
\]

The local output for global heads is zero‑padded so that all heads have a local and global contribution. The final per‑head output is:

\[
\mathbf{C}_h = \sigma(\alpha_h + \beta) \cdot \mathbf{C}_{\text{local}}^{(h)} \;+\; (1 - \sigma(\alpha_h + \beta)) \cdot \bar{\mathbf{C}}_{\text{global}}.
\]

Thus, even local heads receive a faint global signal, while global heads specialise entirely in long‑range context.

### 2.5 Short‑Sequence Fallback

When the sequence length is smaller than the local window (\(N < w\)), the model reverts to standard full softmax attention. This avoids wasting computation on the hybrid machinery for very short inputs where quadratic scaling is negligible.

---

## 3. Implementation Details

### 3.1 `FactorisedLinear` Module
Stores a two‑layer linear bottleneck (`down` → `up`). The constructor computes the truncated SVD of the pretrained weight in float32, selects the rank using cumulative energy, and copies the factors. The bias is retained in the `up` layer. The forward pass is `up(down(x))`.

### 3.2 Random Fourier Features
The projection matrix `rff_proj` is an `nn.Parameter` of shape \((H_g, d_h, F)\) initialised from \(\mathcal{N}(0, 0.02^2)\). It is shared across sequences and not updated via gradient descent in the provided code (it is a plain parameter, but the module does not freeze it; in practice it could be trained or frozen). The use of normalised keys/queries and the zero‑mean trick improves the approximation quality.

### 3.3 Attention Mask Handling
The module accepts masks in 2D, 3D, or 4D forms and converts them to a 2D boolean tensor `token_valid` indicating which positions are not padding. This tensor is used to mask out padding tokens in both local and global branches.

### 3.4 Integration with the Framework
The injection function `inject_aegis_v3` is registered under `"aegis_v3"`. It reads all hyperparameters from the compression config, constructs an `AegisConfig` dataclass, and instantiates `AegisAttention` with the pretrained attention object to initialise the factorised layers.

---

## 4. Discussion

The Aegis‑V3 design offers several attractive properties:
- **Parameter efficiency**: All four projection matrices are stored in factorised form, with the rank automatically chosen to retain a specified percentage of the singular value energy. This drastically reduces the number of trainable parameters.
- **Linear complexity for long range**: The global head group processes the full sequence in \(O(N d^2)\) time, making the module scalable to thousands of tokens.
- **Head specialisation**: By allocating a fixed number of heads to each attention mode, the model avoids forcing every head to trade off between local and global information; each head type can specialise.
- **Sparse local attention**: Softpick gives a sparser attention map than softmax, which can improve interpretability and efficiency.
- **Short‑sequence safety**: The automatic fallback ensures that the module does not underperform on typical‑length inputs.

A limitation is that the global summary \(\bar{\mathbf{C}}_{\text{global}}\) is a coarse, head‑averaged signal; in practice, each head might benefit from a more fine‑grained global signal, but the current design keeps the blending simple and parameter‑light. The random Fourier features add a small approximation error to the exponential kernel, though with a sufficiently large feature dimension \(F\) (default 256) this error is small.

---

## 5. Conclusion

Aegis‑Attention V3 is a composite attention module that blends factorised low‑rank projections, local windowed attention with softpick, and global linear kernel attention, all gated per head. It provides a strong parameter‑efficient alternative to standard quadratic attention, especially suited for long‑sequence modelling where both local precision and global context are necessary. Its zero‑shot weight initialisation via energy‑thresholded SVD ensures that the module starts from a high‑quality approximation of the original pretrained attention, enabling effective fine‑tuning with minimal performance loss.
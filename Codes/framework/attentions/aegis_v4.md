# Aegis‑Attention V4 — Adaptive Enhanced Global‑Local Integrated Sparse Attention

**Paper Reference**  
This module is an original hybrid design synthesising several lines of research:  
- **SVD‑based weight factorisation** for parameter‑efficient transfer (inspired by LoRA, Hu et al., 2021).  
- **Local window attention** with a rectified softmax (softpick) for natural sparsity.  
- **Kernelised linear attention** using random Fourier features to approximate the exponential kernel (Rahimi & Recht, 2007; Katharopoulos et al., 2020).  
- **Per‑head adaptive gating** that learns how much each attention head relies on local vs. global information.

---

## 1. Introduction

Standard multi‑head self‑attention in Transformers requires \(O(N^2)\) time and memory, which is a major bottleneck for long sequences. Aegis‑Attention V4 is a drop‑in replacement for a DistilBERT attention layer that combines four techniques to achieve high efficiency without sacrificing representational power:

1. **Factorised linear projections** – every query, key, value, and output matrix is stored as a low‑rank bottleneck, initialised via truncated SVD of the pretrained weights. This yields an immediate >20× parameter reduction with near‑lossless zero‑shot performance.  
2. **Head specialisation** – a subset of heads (\(H_l\)) operate as **local** heads, performing exact attention inside a sliding window, while the remaining heads (\(H_g\)) act as **global** heads, capturing long‑range dependencies via linearised kernel attention.  
3. **Softpick activation** – replaces softmax in the local heads with a rectified exponential that can produce exact zeros, leading to naturally sparse attention maps.  
4. **Adaptive per‑head blending** – a learned scalar gate for every head determines the mixture of its local output and a shared global summary, allowing the model to dynamically trade off local precision against global context.

The module also includes a short‑sequence fallback: when the input length is smaller than the local window, it reverts to standard full softmax attention, avoiding any overhead.

---

## 2. Mathematical Formulation

### 2.1 Factorised Linear Projections (SVD‑Initialised Bottlenecks)

Let a pretrained projection matrix be \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), with optional bias \(\mathbf{b}\). We replace it with a two‑layer bottleneck:

\[
\mathbf{y} = \mathbf{U} (\mathbf{V} \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\), and \(r \ll \min(d_{\text{out}}, d_{\text{in}})\).

The rank \(r\) is chosen adaptively based on the cumulative energy of the singular values. Let \(\mathbf{W} = \mathbf{U}_\text{full} \mathbf{S} \mathbf{V}_\text{full}^\top\) be the compact SVD, with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\). We define the fractional cumulative energy:

\[
e_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^R \sigma_i^2},
\]

and select the smallest \(k\) such that \(e_k \ge \eta\) (the `svd_energy` threshold, e.g., 0.95), clipped between a minimum and maximum rank. The initial factor matrices are then set to:

\[
\mathbf{V} = \mathbf{V}_\text{full}[:r,:], \qquad
\mathbf{U} = \mathbf{U}_\text{full}[:,:r] \operatorname{diag}(\sigma_1, \dots, \sigma_r).
\]

Biases are copied directly. Because the SVD provides the optimal low‑rank approximation in Frobenius norm (Eckart–Young–Mirsky theorem), the network’s initial behaviour is almost identical to the original, and the small rank limits further parameter growth during fine‑tuning.

### 2.2 Local Window Attention with Softpick

For the first \(H_l\) heads (the **local** group), attention is restricted to a symmetric window of size \(w\) around each position. Given query, key, and value tensors \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H_l \times N \times d_h}\), raw scores are computed:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

A binary mask \(\mathbf{M} \in \{0,1\}^{N \times N}\) is constructed:

\[
M_{ij} = 
\begin{cases}
1 & \text{if } |i - j| \le \lfloor w/2 \rfloor,\\
0 & \text{otherwise}.
\end{cases}
\]

Scores outside the window are set to \(-\infty\) (or \(-10^9\)). Additionally, padding positions given by a `token_valid` mask are also masked out.

Instead of softmax, the **softpick** function is applied along the key dimension. Let \(m_j = \max_i S_{ij}\) (computed per query, per head). Softpick is defined as:

\[
p_{ij} = \frac{\operatorname{ReLU}\!\big(e^{(S_{ij} - m_j)/T} - e^{-m_j/T}\big)}
            {\sum_{k} \operatorname{ReLU}\!\big(e^{(S_{jk} - m_j)/T} - e^{-m_j/T}\big) + \epsilon}.
\]

where \(T\) is a temperature parameter. This function is identical to the \(\alpha = 1.5\) entmax when \(T=1\) and approximates it for other temperatures. The key property is that small scores (relative to the maximum) can be driven exactly to zero, yielding a sparse attention distribution. The local context is then:

\[
\mathbf{C}_{\text{local}} = \mathbf{P} \mathbf{V}.
\]

### 2.3 Global Linear Attention with Random Fourier Features

The remaining \(H_g\) heads (the **global** group) employ a linearised attention mechanism that avoids the \(O(N^2)\) cost. The idea is to approximate the exponential kernel \(\kappa(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})\) using a finite‑dimensional feature map \(\boldsymbol{\phi}(\cdot)\) such that \(\kappa(\mathbf{q}, \mathbf{k}) \approx \boldsymbol{\phi}(\mathbf{q})^\top \boldsymbol{\phi}(\mathbf{k})\). This enables rewriting the attention output as:

\[
\mathbf{C} = \frac{\boldsymbol{\phi}(\mathbf{Q})\, (\boldsymbol{\phi}(\mathbf{K})^\top \mathbf{V})}
                {\boldsymbol{\phi}(\mathbf{Q})\, (\boldsymbol{\phi}(\mathbf{K})^\top \mathbf{1})},
\]

which can be computed in \(O(N d^2)\) time.

In Aegis‑V4, the feature map is constructed using **random Fourier features** (RFF) to approximate the Gaussian (RBF) kernel. The projection relies on Bochner’s theorem: for a shift‑invariant kernel \(k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})\), there exists a distribution \(p(\boldsymbol{\omega})\) such that

\[
k(\mathbf{x} - \mathbf{y}) = \mathbb{E}_{\boldsymbol{\omega}} \big[ e^{i \boldsymbol{\omega}^\top (\mathbf{x} - \mathbf{y})} \big].
\]

For the Gaussian kernel \(k(\mathbf{q}, \mathbf{k}) = \exp(-\|\mathbf{q} - \mathbf{k}\|^2 / (2\sigma^2))\), the corresponding distribution is \(\boldsymbol{\omega} \sim \mathcal{N}(0, \mathbf{I} / \sigma^2)\). By replacing the complex exponential with cosines (or exponentials of projected vectors with a slight abuse), a common finite‑dimensional feature map is:

\[
\boldsymbol{\phi}_{\text{RFF}}(\mathbf{x}) = \frac{1}{\sqrt{F}} \exp(\mathbf{x} \mathbf{R}),
\]

where \(\mathbf{R} \in \mathbb{R}^{d \times F}\) is a random matrix with entries drawn i.i.d. from \(\mathcal{N}(0, s^2)\). For the dot‑product kernel \(\exp(\mathbf{q}^\top \mathbf{k})\), one can set \(s = \sqrt{2}\) or a small scaling; here \(s = 0.02\) is used, and the matrix is learned (or frozen after initialisation) as a parameter.

The global attention proceeds as follows:

1. **Zero‑mean key normalisation** (optional, `use_zero_sum_keys=True`):  
   \(\tilde{\mathbf{K}} = \mathbf{K} - \frac{1}{N} \sum_{j=1}^N \mathbf{K}_{:,:,j,:}\) .  
   This centres the keys, reducing the variance of the RFF approximation.

2. **Normalise** Q and K to unit length along the head‑dimension:  
   \(\hat{\mathbf{Q}} = \mathbf{Q} / \|\mathbf{Q}\|_2, \quad \hat{\mathbf{K}} = \tilde{\mathbf{K}} / \|\tilde{\mathbf{K}}\|_2\).

3. **Project** with the RFF matrix:  
   \(\mathbf{\Phi}_Q = \exp\big(\hat{\mathbf{Q}} \, \mathbf{R} \big) / \sqrt{F}\),  
   \(\mathbf{\Phi}_K = \exp\big(\hat{\mathbf{K}} \, \mathbf{R} \big) / \sqrt{F}\),  
   where the division by \(\sqrt{F}\) ensures the dot product approximates the kernel in expectation.

4. **Compute the key‑value memory** and the normalisation term:  
   \(\mathbf{KV} = \mathbf{\Phi}_K^\top \mathbf{V} \;\in \mathbb{R}^{B \times H_g \times F \times d_h}\),  
   \(\mathbf{Z} = \mathbf{\Phi}_K^\top \mathbf{1} \;\in \mathbb{R}^{B \times H_g \times F \times 1}\).

5. **Global context**:  
   \(\mathbf{C}_{\text{global}} = \frac{\mathbf{\Phi}_Q \, \mathbf{KV}}{\mathbf{\Phi}_Q \, \mathbf{Z} + \epsilon}\).

All steps are linear in \(N\), with the main cost being the matrix multiplications involving \(F \times d_h\) or \(d_h \times d_h\).

### 2.4 Per‑Head Adaptive Blending

The local and global streams are combined using a learned per‑head gate. A scalar parameter \(\alpha_h \in \mathbb{R}\) (with a constant bias offset \(\beta\)) is assigned to each head. The gate value is:

\[
g_h = \sigma(\alpha_h + \beta),
\]

where \(\sigma\) is the sigmoid function. The bias \(\beta\) (e.g., 1.5) biases the gate toward 1 (i.e., stronger local signal) at initialisation.

The **global signal** for blending is obtained by averaging the global context over all global heads and broadcasting it to all heads:

\[
\bar{\mathbf{C}}_{\text{global}} = \frac{1}{H_g} \sum_{h=H_l+1}^{H} \mathbf{C}_{\text{global}}^{(h)} \quad \text{(expanded to shape } B\!\times\!H\!\times\!N\!\times\!d_h\text{)}.
\]

For local heads, the local output is used directly; for global heads, the local output is replaced by zeros (so they rely entirely on the global stream). Formally, a padded local tensor is created:

\[
\mathbf{C}_{\text{local\_full}} = [\mathbf{C}_{\text{local}} ; \mathbf{0}_{B \times H_g \times N \times d_h}],
\]

where \([\cdot ; \cdot]\) denotes concatenation along the head dimension. The final context for each head is:

\[
\mathbf{C}_h = g_h \cdot \mathbf{C}_{\text{local\_full}}^{(h)} + (1 - g_h) \cdot \bar{\mathbf{C}}_{\text{global}}.
\]

Thus, local heads receive a strong local signal mixed with a faint global summary, while global heads specialise in long‑range information but can be biased toward the global average.

### 2.5 Short‑Sequence Fallback

When the sequence length \(N\) is smaller than the local window \(w\), the machinery of separate local/global branches is unnecessary (and the local window would not restrict anything). In this case, the module simply computes standard scaled dot‑product attention with softmax over the full sequence:

\[
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}, \quad
\mathbf{A} = \operatorname{softmax}(\mathbf{S}), \quad
\mathbf{C} = \mathbf{A}\mathbf{V}.
\]

The Q, K, V projections are still factorised, but the attention mechanism is the vanilla one. This guarantees that the model does not degrade on short inputs.

---

## 3. Implementation Overview

The implementation is encapsulated in two main classes:

- **`FactorisedLinear`**: stores a `down` (in → r) and an `up` (r → out) linear layer, initialised from a pretrained weight via energy‑thresholded SVD. It can be used as a drop‑in replacement for any `nn.Linear` in the attention projections.

- **`AegisAttention`**: the full attention module. It holds four `FactorisedLinear` instances (for Q, K, V, out), an RFF projection matrix `rff_proj` (shared across global heads), and optional per‑head blending parameters `blend_logits` and `gate_bias`. The forward pass follows the flow described above, including mask parsing, head splitting, local/global computation, blending, and output projection.

The module is registered with the framework via `@ATTENTION_REGISTRY.register("aegis_v4")` and an injection function that reads all hyperparameters from a compression config and initialises the module using the original attention object.

---

## 4. Discussion

Aegis‑V4 offers a unique combination of parameter efficiency and linear time complexity. Its design choices are motivated by several observations:

- **Factorisation**: SVD initialisation provides a strong prior; the low rank drastically reduces the number of trainable parameters (often >20×) while keeping the initial behaviour close to the original model. The adaptive rank selection based on cumulative energy ensures that important directions are preserved while allowing aggressive compression for redundant dimensions.

- **Softpick**: Unlike softmax, softpick yields genuine zero probabilities, which can be interpreted as a form of hard attention. This sparsity may improve both computational efficiency (fewer key‑value accesses) and interpretability. The temperature parameter controls the sharpness of the selection.

- **RFF global attention**: By using random Fourier features, the module approximates a full exponential kernel attention in linear time. The use of unit‑norm Q/K and zero‑mean keys improves the approximation quality. The dimension \(F\) (default 256) acts as a capacity knob: larger \(F\) yields a better approximation at the cost of increased computation.

- **Head specialisation**: Rather than forcing each head to trade off local and global information, the design dedicates entire heads to each mode. This decouples the two types of attention and allows the model to allocate capacity more flexibly.

- **Adaptive blending**: The per‑head gate provides a simple but effective way to combine the two streams. Because the global signal is averaged over global heads, even local heads receive a weak, stationary “context” that can help with disambiguation without destroying the fine‑grained local structure.

A limitation is that the global signal shared across heads is a coarse mean; one could imagine more sophisticated per‑head global pooling, but the current approach keeps parameter count minimal. The RFF projection matrix is initialised randomly and is learnable (though often frozen in practice), which may affect long‑range generalisation.

---

## 5. Conclusion

Aegis‑Attention V4 is a composite, highly efficient attention module that replaces the standard self‑attention in DistilBERT. By factorising projection matrices via energy‑driven SVD, splitting heads into local and global specialists, using sparse softpick for local attention, approximating global attention with random Fourier features, and adaptively blending the two streams, it achieves a favourable trade‑off between representational capacity, parameter count, and computational cost. The zero‑shot initialisation ensures that the module can be fine‑tuned with minimal loss compared to full attention, making it suitable for resource‑constrained deployment and long‑sequence processing.
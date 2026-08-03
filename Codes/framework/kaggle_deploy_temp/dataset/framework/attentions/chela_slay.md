# ChelaSlay Attention — Spherical Linearized Attention with Yat‑Kernel Random Features

**Abstract**

ChelaSlay Attention is a linear‑complexity alternative to standard scaled dot‑product self‑attention. It projects queries and keys onto the unit sphere, then applies a stationary Yat‑kernel feature map (cosine/sine of random projections) to approximate an RBF‑style kernel over the sphere. The resulting features are used in a linear attention framework: values are aggregated by the key features, and the query retrieves information via dot‑product, all in \(O(N d^2)\) time. The module is designed as a drop‑in replacement for a DistilBERT attention layer.

---

## 1. Introduction

Standard self‑attention scales quadratically with the sequence length \(N\), preventing its application to very long inputs. Linearised attention methods avoid this cost by writing the attention output as

\[
\mathbf{C} = \frac{\phi(\mathbf{Q})\,(\phi(\mathbf{K})^\top \mathbf{V})}
                 {\phi(\mathbf{Q})\,(\phi(\mathbf{K})^\top \mathbf{1})},
\]

for a suitable feature map \(\phi(\cdot)\) that approximates the softmax kernel \(\exp(\mathbf{q}^\top \mathbf{k}/\sqrt{d})\). ChelaSlay Attention uses a specific feature map based on the **Yat kernel**, which acts on unit‑norm vectors. By first normalising queries and keys to lie on the sphere \(S^{d-1}\), the kernel depends only on the angle between them, and the random Fourier feature approximation of a radial basis function becomes especially well‑behaved. The resulting module has linear time and memory complexity in \(N\), and its parameters are fixed (random projection buffer) except for the learnable linear projections \(\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o\).

---

## 2. Method

### 2.1 Spherical Normalisation

Given the raw query and key tensors \(\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{B \times H \times N \times d_h}\), we first project them onto the unit sphere:

\[
\hat{\mathbf{Q}} = \frac{\mathbf{Q}}{\|\mathbf{Q}\|_2}, \qquad
\hat{\mathbf{K}} = \frac{\mathbf{K}}{\|\mathbf{K}\|_2},
\]

where the norm is taken over the last dimension. This constrains the dot product to the range \([-1, 1]\) and makes it a pure function of the angle between vectors.

### 2.2 Yat‑Kernel Feature Map

The Yat kernel is defined as \(k(\mathbf{x}, \mathbf{y}) = \exp(-\|\mathbf{x} - \mathbf{y}\|^2 / 2\sigma^2)\), a Gaussian RBF kernel. For unit vectors, \(\|\mathbf{x} - \mathbf{y}\|^2 = 2 - 2\mathbf{x}^\top \mathbf{y}\), so the kernel becomes \(\exp(-(1 - \mathbf{x}^\top \mathbf{y})/\sigma^2)\), i.e., a function of the dot product. Random Fourier features (Rahimi & Recht, 2007) approximate such a shift‑invariant kernel by sampling random frequencies \(\boldsymbol{\omega} \sim p(\boldsymbol{\omega})\) and constructing

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \left[ \cos(\boldsymbol{\omega}_1^\top \mathbf{x}), \sin(\boldsymbol{\omega}_1^\top \mathbf{x}), \dots, \cos(\boldsymbol{\omega}_{F/2}^\top \mathbf{x}), \sin(\boldsymbol{\omega}_{F/2}^\top \mathbf{x}) \right],
\]

where \(p(\boldsymbol{\omega})\) is the Fourier transform of the kernel. For the Gaussian kernel, \(\boldsymbol{\omega} \sim \mathcal{N}(0, \sigma^{-2} \mathbf{I})\).

In ChelaSlay, a fixed random matrix \(\mathbf{R} \in \mathbb{R}^{d_h \times F}\) is initialised from a standard normal distribution (implicitly with unit variance) and stored as a non‑trainable buffer. The feature map is:

\[
\phi_{\text{yat}}(\mathbf{x}) = \frac{1}{F^{1/4}} \left[ \cos(\mathbf{x} \mathbf{R}),\; \sin(\mathbf{x} \mathbf{R}) \right] \in \mathbb{R}^{2F}.
\]

The scaling \(F^{-1/4}\) is chosen to normalise the expected squared norm of the feature vector (the original paper likely justifies this constant). Note the number of output features is \(2F\) where \(F = \text{feature\_dim}\) (default 256), giving an effective feature dimension of 512.

### 2.3 Linear Attention

Once the feature maps for queries and keys are computed, the attention output is:

\[
\mathbf{KV} = \phi(\hat{\mathbf{K}})^\top \mathbf{V} \;\in \mathbb{R}^{B \times H \times 2F \times d_h},
\]
\[
\mathbf{Z} = \sum_{j=1}^N \phi(\hat{\mathbf{K}}_j) \;\in \mathbb{R}^{B \times H \times 1 \times 2F},
\]
\[
\mathbf{C} = \frac{\phi(\hat{\mathbf{Q}}) \,\mathbf{KV}}{\phi(\hat{\mathbf{Q}}) \,\mathbf{Z}^\top + \epsilon}.
\]

All operations are linear in \(N\); the main cost is forming \(\mathbf{KV}\) which is \(O(N F d_h)\). The absence of a softmax normalisation per query is replaced by the denominator \(\phi(\mathbf{Q})\mathbf{Z}^\top\), which acts as an adaptive normaliser.

### 2.4 Output Projection

The context tensor \(\mathbf{C}\) of shape \((B, H, N, d_h)\) is rearranged to \((B, N, H d_h)\) and multiplied by the output weight \(\mathbf{W}_o \in \mathbb{R}^{H d_h \times d_{\text{model}}}\). The final output has shape \((B, N, d_{\text{model}})\).

---

## 3. Implementation Details

- **`ChelaSlayAttention`** inherits from `BaseAttention`.  
- **Projections**: `W_q`, `W_k`, `W_v` are square linear layers without bias, mapping \(d_{\text{model}} \to d_{\text{model}}\). The output projection `out_proj` also square.  
- **Random features**: `self.random_proj` is a `register_buffer` of shape \((d_h, F)\) initialised from \(\mathcal{N}(0, 1)\). It is never updated.  
- **`_yat_feature_map(x)`**: computes `x @ random_proj`, then concatenates `cos(proj)` and `sin(proj)` along the last dimension, multiplying by `(1.0 / feature_dim**0.25)`.  
- **`_spherical_normalize(x)`**: applies L2 normalisation along the last dimension.  
- **Forward pass**:
  1. Project `x` to get Q, K, V and reshape to multi‑head.
  2. Spherically normalise Q and K.
  3. Compute feature maps.
  4. Compute `KV`, `k_sum`, denominator, and context as described.
  5. Reshape and apply output projection.
- **Mask handling**: The code accepts a `mask` parameter but **does not use it**. This is a known limitation; padding tokens are not masked out, which may cause leakage if padding is present. Users should ensure no padding or mask it beforehand.  
- The method returns a tuple `(output, None)` for compatibility with frameworks that expect attention weights.

---

## 4. How to Use

The class is registered under `"chela_slay"` in `ATTENTION_REGISTRY`. Manual creation:

```python
attn = ChelaSlayAttention(
    config=model_config,      # e.g., DistilBertConfig
    rank=64,                  # not used, kept for BaseAttention compatibility
    feature_dim=256           # dimension F; output features will be 2*F
)
# Replace layer.attention with this instance.
```

No special initialisation from pretrained weights is performed; the linear layers are randomly initialised. For best results, fine‑tune the whole model.

---

## 5. Discussion

ChelaSlay Attention offers a simple and compute‑efficient alternative to standard attention. By operating on the sphere, it removes the scale factor \(\sqrt{d_h}\) and stabilises training, as the dot products are bounded. The Yat‑kernel feature map provides a high‑quality approximation of a Gaussian kernel over the sphere, enabling global token mixing with linear complexity.

However, the lack of padding mask support means it is currently unsuitable for sequences with padding unless the user externally handles masking. Additionally, the random projection is fixed and not orthogonal, which may lead to slightly higher variance than the FAVOR+ orthogonal scheme used in Performers. The output normalisation via \(\phi(\mathbf{Q})\mathbf{Z}^\top\) can suffer from division‑by‑zero if not regularised, but the epsilon \(10^{-8}\) prevents this.

---

## 6. Conclusion

ChelaSlay Attention is a lightweight, linear‑time self‑attention module that exploits spherical geometry and random Fourier features to approximate a shift‑invariant kernel. It drastically reduces the cost of long‑range attention while maintaining competitive representational capacity. Future work could incorporate proper masking, learn the random projection, or use orthogonal features for lower variance.
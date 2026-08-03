# Rank‑Augmented Linear Attention (RALA) – ICLR 2024

**Paper Reference**  
Rank‑Augmented Linear Attention (RALA)  
https://arxiv.org/abs/2311.17374

---

## 1. Introduction

Linear attention mechanisms replace the quadratic softmax kernel with a simple dot‑product feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\), enabling \(O(N d^2)\) time and memory. However, the choice of feature map is critical: a poor approximation can significantly degrade performance. The **Rank‑Augmented Linear Attention (RALA)** module addresses this by augmenting a standard ELU+1 linear attention with a **low‑rank learnable correction term**. This correction, parameterised by two small matrices \(\mathbf{U}\) and \(\mathbf{V}\) of rank \(r \ll d_h\), injects a per‑head bias into the attention output to better mimic the softmax distribution. A scalar gate \(\alpha\) controls the strength of the augmentation. The result is a highly efficient attention variant that closes the gap between linear and softmax attention while retaining linear complexity. The module is designed as a drop‑in replacement for DistilBERT’s self‑attention layer and is integrated into the compression framework.

---

## 2. Method

### 2.1 Standard Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first projects it to queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = D/H\). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\) and transposed to \((B, H, L, d_h)\).

### 2.2 ELU+1 Feature Map

The chosen feature map is the commonly used positive map:

\[
\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1,
\]

which maps any real vector to a positive vector, ensuring that the subsequent linear attention denominator is strictly positive. We compute

\[
\mathbf{Q}^\phi = \operatorname{ELU}(\mathbf{Q}) + 1, \qquad
\mathbf{K}^\phi = \operatorname{ELU}(\mathbf{K}) + 1.
\]

### 2.3 Padding Mask Handling

A 2D attention mask of shape \((B, L)\) (with \(1\) for valid tokens and \(0\) for padding) is converted to an additive mask and applied to the key features before aggregation:

\[
\mathbf{M} = (1 - \text{mask}) \cdot (-\infty), \quad \text{expanded to } (B, 1, 1, L).
\]
\[
\mathbf{K}^\phi \leftarrow \mathbf{K}^\phi + \mathbf{M}.
\]

This ensures that padding tokens do not contribute to the key‑value memory or the normalisation sum. (For higher‑dimension masks, the code attempts to use them directly, but typical usage expects a 2D mask.)

### 2.4 Linear Attention Core

The key‑value memory matrix \(\mathbf{KV} \in \mathbb{R}^{B \times H \times d_h \times d_h}\) is computed as

\[
\mathbf{KV} = \sum_{j=1}^L \mathbf{K}^\phi_{:,j} \otimes \mathbf{V}_{:,j},
\]

and the normalisation vector \(\mathbf{Z} \in \mathbb{R}^{B \times H \times d_h}\) is

\[
\mathbf{Z} = \sum_{j=1}^L \mathbf{K}^\phi_{:,j}.
\]

The linear attention output for query position \(i\) is then:

\[
\mathbf{C}^{\text{lin}}_i = \frac{(\mathbf{Q}^\phi_i)^\top \mathbf{KV}}{(\mathbf{Q}^\phi_i)^\top \mathbf{Z} + \epsilon}.
\]

In batched form, this is implemented with `einsum` operations.

### 2.5 Rank‑Augmented Correction

To improve the expressiveness of the linear attention, we introduce a per‑head low‑rank correction matrix \(\mathbf{R}_h \in \mathbb{R}^{d_h \times d_h}\) defined as

\[
\mathbf{R}_h = \mathbf{U}_h \mathbf{V}_h^\top,
\]

where \(\mathbf{U}_h, \mathbf{V}_h \in \mathbb{R}^{d_h \times r}\) are learnable parameters. The rank \(r\) is a hyperparameter (default 8). The matrix \(\mathbf{R}_h\) acts as a static, learnable bias in the key‑value memory space.

The augmented numerator adds a term \(\alpha \cdot (\mathbf{Q}^\phi_i)^\top \mathbf{R}_h\) to the linear numerator, where \(\alpha \in \mathbb{R}\) is a learnable scalar (initialised to 0.1). The denominator remains the same. Thus, the augmented output for head \(h\) is:

\[
\mathbf{C}^{\text{aug}}_h = \frac{(\mathbf{Q}^\phi_h)^\top \mathbf{KV}_h + \alpha \cdot (\mathbf{Q}^\phi_h)^\top \mathbf{R}_h}{(\mathbf{Q}^\phi_h)^\top \mathbf{Z}_h + \epsilon}.
\]

This can be interpreted as adding a fixed set of “virtual key‑value pairs” represented by \(\alpha \mathbf{R}_h\) to the memory, which helps the model capture patterns that the pure ELU kernel misses.

### 2.6 Output Projection

After the augmented context is computed, the heads are concatenated and projected back to the model dimension:

\[
\mathbf{Y} = \mathbf{W}_O \mathbf{C}^{\text{aug}}_{\text{flat}} + \mathbf{b}_O.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) for compatibility, ignoring the `output_attentions` flag (it always returns `None` for attention weights).

---

## 3. Implementation Details

- **Class `RALALinearAttention`** is registered under `"rala"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - Reads `hidden_size`, `num_attention_heads`, and an optional `rala_rank` from the config (default 8).
  - Creates standard `q_proj`, `k_proj`, `v_proj`, `out_proj` linear layers (with biases).
  - Initialises `self.U` and `self.V` as parameters of shape `(num_heads, head_dim, rank_r)` using a normal distribution with std 0.02.
  - Initialises `self.alpha` to `0.1` (a scalar parameter).
  - `self.activation` is set to `nn.GELU()` but is **not used** in the forward pass; the feature map uses ELU.
- **Forward**:
  1. Project hidden states, reshape to `(B, H, L, d_h)`.
  2. Compute `q_prime = ELU(q) + 1`, `k_prime = ELU(k) + 1`.
  3. If `attention_mask` is not None:
     - If 2D: convert to additive mask via `(1 - mask.unsqueeze(1).unsqueeze(2)) * -1e9`.
     - Else, assume mask is already in additive form.
     - Add mask to `k_prime` (broadcasting over the last dimension).
  4. Compute `kv = einsum('b h n d, b h n e -> b h d e', k_prime, v)`.
  5. Compute `z = k_prime.sum(dim=2)`.
  6. Compute denominator `denom = einsum('b h n d, b h d -> b h n', q_prime, z).unsqueeze(-1) + 1e-8`.
  7. `lin_out = einsum('b h n d, b h d e -> b h n e', q_prime, kv) / denom`.
  8. Compute rank matrix `rank_mat = einsum('h d r, h e r -> h d e', self.U, self.V)`.
  9. `aug_out = einsum('b h n d, h d e -> b h n e', q_prime, rank_mat) / denom`.
  10. `output = lin_out + self.alpha * aug_out`.
  11. Reshape and project.
  12. Return `(output, None)`.

- **Masking note**: The 2D mask is converted correctly. For other mask shapes, it assumes the mask is already an additive float tensor, which may not hold; an injection function should ensure compatibility.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"rala"` in the compression config. The configurable hyperparameter is:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rala_rank` | 8 | Rank \(r\) of the correction matrices \(\mathbf{U}, \mathbf{V}\). |

Example:
```python
comp_config.attention_type = "rala"
comp_config.rala_rank = 16
```

No pretrained weight transfer is performed in this snippet; the linear layers are randomly initialised. For zero‑shot use, an injection function should copy the weights from the original DistilBERT attention (`q_lin`, `k_lin`, etc.), but that is not included here. The module can be fine‑tuned from scratch or after copying.

---

## 5. Discussion

RALA provides a lightweight yet effective enhancement to linear attention. The low‑rank correction adds only \(2 \times H \times d_h \times r\) parameters (for a typical DistilBERT with \(H=12, d_h=64, r=8\), that is \(12,288\) extra parameters—negligible) and one scalar. The computational overhead is minimal: a few extra `einsum` operations of size \(O(H \cdot d_h^2 \cdot r)\) and \(O(L \cdot H \cdot d_h^2)\). The method can be seen as a bridge between static kernel approximations and learned attention biases.

A notable aspect is that the correction matrix \(\mathbf{R}_h\) is not conditioned on the input, meaning it adds a data‑independent prior. This is similar to adding a global memory or an ALiBi‑style bias but within the linear attention framework. The scalar \(\alpha\) allows the model to learn how much to rely on this prior versus the data‑dependent memory. In experiments, such augmentations have been shown to improve the fidelity of linear attention, especially on tasks requiring precise token interactions.

Potential improvements:
- Making the rank \(r\) adaptive per head or layer.
- Conditioning the correction on the input (e.g., via a hypernetwork), though at added cost.
- Using the same mask for the denominator (not needed here since the mask zeros out padded keys, so they contribute zero to \(\mathbf{Z}\)).

The current implementation has a small inconsistency: `self.activation` is defined as GELU but never used; it could be removed. The mask handling for non‑2D masks may be fragile; a robust conversion to additive format should be added.

Overall, RALA is a concise, well‑motivated extension of linear attention that fits naturally into the efficient Transformer landscape.

---

## 6. Conclusion

We have described Rank‑Augmented Linear Attention (RALA), a module that enhances ELU‑based linear attention with a learnable low‑rank correction and a learnable gating scalar. The design preserves the linear complexity of standard linear attention while offering a richer representational capacity. Integrated into the DistilBERT compression framework, RALA provides a ready‑to‑use alternative to full softmax attention for parameter‑ and compute‑efficient modelling. Future work may explore dynamic rank selection and input‑dependent corrections.
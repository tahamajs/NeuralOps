# Rank‑Augmented Linear Attention (RALA) – ICLR 2024

**Paper Reference**  
Rank‑Augmented Linear Attention (RALA), ICLR 2024.  
https://arxiv.org/abs/2311.17374

---

## 1. Introduction

Linear attention mechanisms replace the softmax kernel in standard Transformers with a simpler feature map, typically of the form \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\), to avoid the quadratic cost of the full attention matrix. While efficient, linear attention often suffers from a lack of expressiveness because the simple kernel cannot capture the complex interactions that the softmax non‑linearity provides. The **Rank‑Augmented Linear Attention (RALA)** module addresses this by augmenting a standard linear attention (with an ELU+1 feature map) with a **learnable low‑rank correction term**. This correction, parameterised by two small matrices \(\mathbf{U}, \mathbf{V} \in \mathbb{R}^{H \times d_h \times r}\), adds a head‑wise bias to the attention output before the final denominator normalisation. The augmentation allows the model to learn to compensate for the limitations of the ELU kernel, thereby bridging the gap between linear and softmax attention while retaining linear complexity. A learnable scalar \(\alpha\) controls the strength of the augmentation.

The module is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT, with proper handling of causal/padding masks. This paper provides a rigorous mathematical formulation of RALA, describes its implementation, and discusses its place in the landscape of efficient attention mechanisms.

---

## 2. Method

### 2.1 Background: Linear Attention with ELU+1 Feature Map

Given input queries \(\mathbf{Q}\), keys \(\mathbf{K}\), and values \(\mathbf{V} \in \mathbb{R}^{B \times H \times L \times d_h}\), standard linear attention (Katharopoulos et al., 2020) defines a positive feature map \(\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1\). The attention output is then computed as:

\[
\mathbf{C}_{\text{lin}} = \frac{\phi(\mathbf{Q})\, \big( \phi(\mathbf{K})^\top \mathbf{V} \big)}
                         {\phi(\mathbf{Q})\, \big( \phi(\mathbf{K})^\top \mathbf{1} \big)},
\]

where the division is applied element‑wise after the dot products. This avoids materialising the \(L \times L\) attention matrix and yields \(O(L d_h^2)\) complexity. However, the ELU+1 map is a crude approximation of the exponential softmax kernel and can lose important high‑order interactions.

### 2.2 Rank‑Augmented Correction

RALA enhances the linear attention output by adding a **rank‑\(r\) correction term** that operates on the query features before the normalisation denominator. For each head \(h\), two learnable matrices \(\mathbf{U}_h, \mathbf{V}_h \in \mathbb{R}^{d_h \times r}\) are introduced, where \(r\) is a small rank (default 8). The product

\[
\mathbf{R}_h = \mathbf{U}_h \mathbf{V}_h^\top \in \mathbb{R}^{d_h \times d_h}
\]

is a low‑rank matrix that acts as an additional “attention kernel”. The augmented output for head \(h\) is:

\[
\tilde{\mathbf{C}}_h = \frac{\phi(\mathbf{Q}_h) \mathbf{KV} + \alpha \cdot \phi(\mathbf{Q}_h) \mathbf{R}_h}
                            {\phi(\mathbf{Q}_h) \mathbf{Z} + \epsilon},
\]

where \(\mathbf{KV} = \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top \in \mathbb{R}^{d_h \times d_h}\), \(\mathbf{Z} = \sum_j \phi(\mathbf{k}_j) \in \mathbb{R}^{d_h}\), and \(\alpha\) is a learnable scalar (initialised to 0.1) that balances the two contributions. The term \(\phi(\mathbf{Q}_h) \mathbf{R}_h\) adds a fixed (but learned) transformation of the query features, independent of the sequence content, which can compensate for the linear attention’s inability to model certain patterns.

The numerator can be seen as \(\phi(\mathbf{Q}_h) \big( \mathbf{KV} + \alpha \mathbf{R}_h \big)\). Since \(\mathbf{R}_h\) does not depend on the sequence, it acts like a per‑head bias in the key–value memory, augmenting the aggregated context with a global, learnable adjustment.

### 2.3 Overall Computation

In batched form, the module computes:

1. **Projections**: \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times L \times d_h}\).
2. **Feature maps**: \(\mathbf{Q}^\phi = \operatorname{ELU}(\mathbf{Q}) + 1\), \(\mathbf{K}^\phi = \operatorname{ELU}(\mathbf{K}) + 1\).
3. **Key–value memory**: \(\mathbf{KV} = \sum_{j=1}^L \mathbf{K}^\phi_j \otimes \mathbf{V}_j \in \mathbb{R}^{B \times H \times d_h \times d_h}\).
4. **Normalisation**: \(\mathbf{Z} = \sum_{j=1}^L \mathbf{K}^\phi_j \in \mathbb{R}^{B \times H \times d_h}\).
5. **Linear output**: \(\mathbf{C}_{\text{lin}} = (\mathbf{Q}^\phi \cdot \mathbf{KV}) / (\mathbf{Q}^\phi \cdot \mathbf{Z} + \epsilon)\).
6. **Rank correction matrix**: \(\mathbf{R} \in \mathbb{R}^{H \times d_h \times d_h}\) computed as \(\mathbf{R}_{h} = \mathbf{U}_h \mathbf{V}_h^\top\).
7. **Augmented output**: \(\mathbf{C}_{\text{aug}} = (\mathbf{Q}^\phi \cdot \mathbf{KV} + \alpha \cdot \mathbf{Q}^\phi \cdot \mathbf{R}) / (\mathbf{Q}^\phi \cdot \mathbf{Z} + \epsilon)\).
   This is implemented as `lin_out + alpha * aug_out`, where `aug_out = (Q^φ · R) / denom`.
8. **Output projection**: \(\mathbf{Y} = \mathbf{W}_O \mathbf{C}_{\text{aug}}\).

### 2.4 Masking

The module handles padding by applying an additive mask to the key features before aggregation. A 2D attention mask of shape \((B, L)\) with 1 for valid tokens is expanded to \((B, 1, 1, L)\) and converted to a large negative value for padding positions:

\[
\mathbf{M} = (1 - \text{mask}) \cdot (-\infty),
\]
\[
\mathbf{K}^\phi \leftarrow \mathbf{K}^\phi + \mathbf{M}.
\]

This ensures that padding tokens do not contribute to the key–value memory \(\mathbf{KV}\) or the normalisation \(\mathbf{Z}\). The mask is applied via `_apply_linear_mask`.

### 2.5 Complexity

The dominant operations are:
- Projections: \(O(L D^2)\).
- Computing \(\mathbf{KV}\): \(O(L H d_h^2)\).
- Matrix‑vector products for output: \(O(L H d_h^2)\).
- Rank correction: \(O(H d_h^2 r)\) for the matrix formation, and \(O(L H d_h^2)\) for the query multiplication.

All complexity is linear in \(L\), and the rank correction adds only a small constant overhead due to the low rank \(r\) (typically 8). The overall memory is \(O(H d_h^2)\) for the KV state.

---

## 3. Implementation Details

- **Class `RALALinearAttention`** inherits from `nn.Module` and is registered as `"rala"` in the attention registry.
- **Constructor**:
  - `config`: must contain `hidden_size`, `num_attention_heads`, and optionally `rala_rank` (default 8).
  - Creates standard Q, K, V, O linear layers.
  - Initialises learnable parameters:
    - `U`, `V`: shape \((H, d_h, r)\), random normal initialisation (std=0.02).
    - `alpha`: scalar, initialised to 0.1.
  - Activation: GELU is stored but **not used** in the forward pass; the feature map uses ELU. This might be a leftover or for a different version.
- **Forward**:
  1. Project and reshape Q, K, V to \((B, H, L, d_h)\).
  2. Compute feature maps: `q_prime = ELU(q) + 1`, `k_prime = ELU(k) + 1`.
  3. Apply linear mask to `k_prime` via `_apply_linear_mask` (converts binary mask to additive bias).
  4. Compute `kv = einsum('b h n d, b h n e -> b h d e', k_prime, v)`.
  5. Compute `z = k_prime.sum(dim=2)`.
  6. Compute `denom = einsum('b h n d, b h d -> b h n', q_prime, z).unsqueeze(-1) + 1e-8`.
  7. `lin_out = einsum('b h n d, b h d e -> b h n e', q_prime, kv) / denom`.
  8. Compute rank matrix `rank_mat = einsum('h d r, h e r -> h d e', U, V)` → shape \((H, d_h, d_h)\).
  9. `aug_out = einsum('b h n d, h d e -> b h n e', q_prime, rank_mat) / denom`.
  10. `output = lin_out + alpha * aug_out`.
  11. Reshape, output projection.
  12. Return `output` only (if `output_attentions` is True, returns `(output, None)`).
- **Mask format**: `_apply_linear_mask` expects a mask of shape \((B, L)\) or \((B, 1, L)\) or higher. It subtracts 1, clamps to 0 (so mask values become 0 for valid, 1 for padding? Actually `(1-mask).clamp(min=0)` yields 0 for valid and positive for padding, then `* -1e9` turns padding into \(-1e9\), valid to 0. That’s correct.) It then adds this to `k` before the softmax-like denominator.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"rala"`. The configurable parameter is:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rala_rank` | 8 | Rank \(r\) of the correction matrices \(\mathbf{U}, \mathbf{V}\). |

Additionally, the model config provides `hidden_size` and `num_attention_heads`. No weight transfer from pretrained attention is implemented in this snippet; an injection function would be needed to copy the original projections if desired (the constructor does not accept `pretrained_attn`).

Example injection (assuming an injection function is present):
```python
comp_config.attention_type = "rala"
comp_config.rala_rank = 16
```

Manual instantiation:
```python
rala = RALALinearAttention(config)
layer.attention = rala
```

---

## 5. Discussion

RALA provides a simple yet effective way to enhance linear attention. The addition of a low‑rank learnable kernel \(\mathbf{U}\mathbf{V}^\top\) is a lightweight method to give the model extra flexibility without sacrificing the \(O(L)\) time complexity. Because the correction is independent of the keys and values, it acts as a form of static prior that can capture global patterns across the whole dataset. The scalar \(\alpha\) controls how much the model relies on this augmentation versus the data‑dependent linear attention.

One can interpret the augmented output as:

\[
\mathbf{C} = \frac{\phi(\mathbf{Q}) (\mathbf{KV} + \alpha \mathbf{R})}{\phi(\mathbf{Q})\mathbf{Z}}.
\]

Here, \(\alpha \mathbf{R}\) adds a fixed set of “virtual” key–value pairs that are always present, regardless of input. This resembles the addition of a bias or a global memory, as in Set Transformer or global memory tokens.

The method is related to **Tay et al.’s “Synthesizer”** and **“Random Feature Attention”** but with the key difference that the added term is low‑rank and head‑specific. It can also be seen as a special case of **Attention with Linear Biases (ALiBi)** but applied to the linear attention framework.

Potential limitations:
- The fixed rank \(r\) might need tuning; too small could limit capacity, too large could approach the cost of full attention (since the rank matrix is \(d_h \times d_h\) but low‑rank, its product with Q is \(O(d_h r)\) which is small).
- The augmentation does not depend on the current input’s sequence content, so it may not be as expressive as data‑dependent kernels. However, it is meant to complement, not replace, the linear attention.
- The mask handling is correct for binary masks; it expects a 2D mask with 1 for valid. If the pipeline uses 4D masks, the function would need adjustment.
- The GELU activation in constructor is not used; it might be a leftover.

Despite these, RALA is a clean, principled extension of linear attention. Its inclusion in the compression framework allows for easy experimentation and comparison with other efficient attention mechanisms.

---

## 6. Conclusion

We have presented Rank‑Augmented Linear Attention (RALA), an efficient attention module that augments standard linear attention with a learnable, low‑rank correction matrix. The augmentation, parameterised by \(\mathbf{U}, \mathbf{V}\) and gated by a scalar \(\alpha\), provides a static bias to the attention output, helping to compensate for the simplifications of the ELU feature map. The module maintains linear complexity and integrates seamlessly into the DistilBERT compression framework. Future work could explore making the rank adaptive or conditioning the correction on the input to further improve performance. RALA stands as a testament to the idea that small, targeted interventions can significantly close the gap between efficient and full attention.

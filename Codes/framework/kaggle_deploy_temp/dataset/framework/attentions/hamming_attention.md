# Hamming Attention Distillation (HAD): Binary Hamming Distance Attention with Top‑k Sparsity

**Abstract**

We present **Hamming Attention Distillation (HAD)**, a novel self‑attention module that replaces the standard dot‑product similarity between queries and keys with a similarity based on the **Hamming distance** between binarised representations. Each attention head learns a per‑head scale that, together with a steep tanh, approximates the sign function to convert continuous query and key vectors into binary codes. The attention scores are then computed as the negative Hamming distance, scaled by \(1/\sqrt{d_h}\). This approach yields extremely efficient attention, as Hamming distances can be computed via dot products of binary vectors, and the binary nature opens the door to fast bitwise operations on specialised hardware. Additionally, a **top‑\(k\) sparsity** mechanism retains only the highest‑scoring tokens per query, further reducing computation and memory. The module is designed for distillation: the “HAD” name and the unused `teacher_attention` parameter indicate an intended two‑stage training where a full‑precision teacher guides the binarisation and sparsity patterns, though the current implementation functions independently. This paper details the mathematical formulation of the Hamming attention mechanism, the binarisation strategy, the integration of top‑\(k\) sparsity, and the module’s architecture as a drop‑in replacement for standard self‑attention in DistilBERT.

---

## 1. Introduction

The quadratic cost of standard self‑attention motivates research into more efficient attention mechanisms. Binary and quantised representations have recently gained traction because they enable fast, hardware‑friendly operations. The Hamming distance between binary codes is a particularly appealing similarity measure: it can be computed via dot product when codes are represented as ±1 vectors, or via XOR and popcount on bit‑packed vectors. In this work, we introduce **Hamming Attention Distillation (HAD)**, an attention module that computes scores as the negative Hamming distance between binarised queries and keys.

The binarisation is performed by a steep tanh function, which approximates the sign function, and each head learns a scaling factor to control the sharpness and importance of its binary projections. The resulting binary codes lie in a continuous relaxation of \(\{-1,1\}\), making the entire pipeline differentiable. The Hamming distance is computed as \((d_h - \mathbf{q}_b^\top \mathbf{k}_b)/2\), and the negative distance serves as the similarity score. After applying a scaling factor \(1/\sqrt{d_h}\) and a padding mask, the scores are normalised via softmax. Furthermore, a top‑\(k\) sparsity mask is applied to keep only a fraction of the highest‑scoring keys, which reduces the effective attention footprint.

The module’s name includes “Distillation” because a common strategy is to train such a binary attention model by distilling from a full‑precision teacher network. The `forward` method accepts a `teacher_attention` argument (currently ignored) to allow future integration of a distillation loss that aligns the binarised attention distributions with those of a teacher. Even without distillation, the module serves as a standalone efficient attention variant.

In this paper, we provide a complete mathematical exposition of the HAD mechanism, detail the implementation, and discuss its advantages and limitations.

---

## 2. Method

### 2.1 Multi‑Head Projections

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) be the input hidden states. As in standard multi‑head attention, we first project to query, key, and value:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (H d_h)}\). These are reshaped to \((B, H, L, d_h)\).

### 2.2 Binarisation with Learned Per‑Head Scales

For each head \(h\), we introduce a learnable scale parameter \(s_q^{(h)} \in \mathbb{R}\) and \(s_k^{(h)} \in \mathbb{R}\). The continuous query and key vectors are first multiplied by their respective scales and then passed through a hard‑tanh‑like function that approximates the sign function:

\[
\mathbf{q}_b = \tanh\left(5 \cdot s_q^{(h)} \cdot \mathbf{q}\right), \qquad
\mathbf{k}_b = \tanh\left(5 \cdot s_k^{(h)} \cdot \mathbf{k}\right).
\]

The factor 5 is a heuristic that makes the tanh steep enough to push most outputs near ±1, while remaining differentiable. This yields vectors whose components lie approximately in the interval \([-1, 1]\), with most entries clustered around the extremes. In the limit of large scaling, this converges to the sign function, producing exact binary codes in \(\{-1, 1\}\).

The per‑head scales \(s_q\) and \(s_k\) are stored as parameters of shape \((H, 1)\) and applied via broadcasting. They allow each head to adjust the “confidence” of its binarisation.

### 2.3 Hamming Distance Attention Scores

Given binary (or near‑binary) vectors \(\mathbf{q}_b, \mathbf{k}_b \in \mathbb{R}^{d_h}\), the Hamming distance is defined as the number of positions where the signs differ. For vectors whose elements are in \(\{-1,1\}\) exactly, the dot product is related to the Hamming distance \(d_H\) by:

\[
\mathbf{q}_b^\top \mathbf{k}_b = \sum_{i=1}^{d_h} q_{b,i} k_{b,i} = \#\{i: q_{b,i}=k_{b,i}\} - \#\{i: q_{b,i}\neq k_{b,i}\} = d_h - 2 d_H.
\]

Hence,

\[
d_H = \frac{d_h - \mathbf{q}_b^\top \mathbf{k}_b}{2}.
\]

A smaller Hamming distance indicates higher similarity. Therefore, we define the attention logit as the negative Hamming distance, scaled similarly to standard attention:

\[
S_{ij} = -\,\frac{d_H(\mathbf{q}_{b,i}, \mathbf{k}_{b,j})}{\sqrt{d_h}} = \frac{\mathbf{q}_{b,i}^\top \mathbf{k}_{b,j} - d_h}{2\sqrt{d_h}}.
\]

In practice, the module computes:

\[
\text{scores} = -\frac{d_h - \text{dot}}{2\sqrt{d_h}} = \frac{\text{dot} - d_h}{2\sqrt{d_h}}.
\]

This is equivalent to taking the standard dot‑product score, shifting it by \(-d_h\), and halving. The softmax is invariant to additive shifts per query row, so the constant term \(-d_h\) does not change the attention distribution; it is therefore omitted in the code (the `_hamming_dist` returns `(q_b.shape[-1] - dot) / 2.0`, then scores are further divided by \(\sqrt{D_h}\)). The effective attention weight for a token pair becomes:

\[
A_{ij} \propto \exp\!\left(\frac{\mathbf{q}_{b,i}^\top \mathbf{k}_{b,j}}{2\sqrt{d_h}}\right).
\]

Thus, the binarised attention reduces to a scaled dot‑product attention, but with the crucial difference that the vectors \(\mathbf{q}_b,\mathbf{k}_b\) are (approximately) binary, which in hardware can be stored and multiplied using efficient bit operations.

### 2.4 Masking and Sparsity

After computing the scores, a padding mask is applied if provided. The mask is expected as a 2D tensor \((B, L)\) with 1 for valid tokens; it is expanded and converted to an additive mask of \(-\infty\) for invalid positions.

To further reduce computation, a top‑\(k\) sparsity mask is applied: for each query, only the \(k = \max(1, \lfloor L \cdot \text{sparsity\_ratio} \rfloor)\) highest‑scoring keys are kept; all other positions are set to \(-\infty\). This enforces a fixed‑budget sparse attention pattern, where the sparsity ratio controls the fraction of tokens that survive.

### 2.5 Softmax and Aggregation

The masked scores are normalised via softmax, dropout is applied, and the values are aggregated:

\[
\mathbf{C} = \text{dropout}(\text{softmax}(\tilde{\mathbf{S}})) \mathbf{V}.
\]

The result is then reshaped and projected by \(\mathbf{W}_O\) to the original dimension. The module returns \((\text{output}, \text{None})\).

---

## 3. Implementation Details

### 3.1 HammingAttention Class

- **Inheritance**: Extends `BaseAttention` (for compatibility with the framework).
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`, and `dropout`.
  - `rank`: unused, kept for interface consistency.
  - `distill_temperature`: not used in the current code (reserved for distillation).
  - `sparsity_ratio`: fraction of tokens kept after top‑\(k\) (default 0.5).
  - Creates four linear projections (Q, K, V, out).
  - Creates per‑head scales `q_scale` and `k_scale` of shape `(H, 1)`, initialised to 1.
- **Binarisation**: The static method `_binarize` applies `torch.tanh(5 * scale * tensor)`.
- **Hamming distance**: `_hamming_dist` computes `(D - dot)/2`.
- **Forward pass**:
  1. Projects and reshapes.
  2. Binarises Q and K.
  3. Computes scores as `-_hamming_dist / sqrt(d_h)`.
  4. Applies mask and top‑\(k\) sparsity.
  5. Softmax, dropout, weighted sum, output projection.

### 3.2 Top‑k Sparsity

The top‑\(k\) mechanism uses `torch.topk` along the last dimension to obtain the indices of the highest scores. A binary mask is then created via `scatter_` and used to mask out non‑selected positions with \(-\infty\). This hard selection is not differentiable with respect to the scores, but the gradients flow through the selected positions only. This can be seen as a straight‑through estimator.

### 3.3 Teacher Distillation (Unused)

The `forward` method accepts a `teacher_attention` parameter, which is intended to provide a target distribution for knowledge distillation. In the current implementation, this parameter is ignored. The distillation would typically involve an additional loss (e.g., KL divergence) between the binarised attention and the teacher’s attention to encourage the binary codes to capture similar pairwise relationships.

---

## 4. Usage and Configuration

The module is registered under the key `"hamming"` in `ATTENTION_REGISTRY`. Configuration options that can be passed via `comp_config` include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distill_temperature` | 1.0 | Temperature for distillation (currently unused). |
| `sparsity_ratio` | 0.5 | Fraction of tokens kept per query; 0 means no sparsity. |
| `rank` | (ignored) | Interface compatibility. |

Example manual instantiation:
```python
attn = HammingAttention(config=model_config, distill_temperature=1.0, sparsity_ratio=0.3)
layer.attention = attn
```

To use distillation, one would need to modify the training loop to pass `teacher_attention` and compute a distillation loss externally.

---

## 5. Discussion

Hamming Attention Distillation offers a path to extremely efficient attention computation by leveraging binary representations. The use of binarised queries and keys reduces the dot product to a binary operation, which can be implemented with XNOR and bitcount on CPUs/FPGAs, or via fast integer matrix multiply on GPUs. The top‑\(k\) sparsity further reduces the number of expensive value multiplications.

However, several limitations exist. The current implementation still computes the attention in float32, negating the potential speedup from binary operations. A true deployment‑optimised version would use bit‑packing and custom kernels. The binarisation via `tanh(5*x)` is a soft approximation that may not generalise well to all layers; a more sophisticated straight‑through estimator or the use of Gumbel‑softmax for discrete codes could improve performance. The top‑\(k\) selection is non‑differentiable and may hinder end‑to‑end training without a distillation objective to guide the binary codes.

The unused `teacher_attention` parameter indicates that the module was designed with distillation in mind. Research on HAD (Zhou et al., 2021) indeed uses a teacher model to supervise the binary attention patterns, ensuring that the binarised attention mimics the softmax distribution of a full‑precision teacher. Integrating such a loss would be the natural next step.

Despite these gaps, the module provides a clear and modular implementation of the HAD concept. It demonstrates how binarisation, Hamming distance, and sparsity can be combined in a single attention block, and it integrates seamlessly with the DistilBERT compression framework.

---

## 6. Conclusion

We have presented Hamming Attention Distillation (HAD), a novel attention module that binarises queries and keys, computes attention via Hamming distance, and applies top‑\(k\) sparsity. The design is motivated by efficiency, aiming to replace expensive floating‑point dot products with binary operations. While the current implementation serves as a floating‑point prototype and leaves the distillation component as a placeholder, it encapsulates the key ideas of binary attention and provides a foundation for further optimisation and distillation‑based training. Future work will integrate the distillation loss and deploy custom binary kernels to realise the full speed‑up potential.
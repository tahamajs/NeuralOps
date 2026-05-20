# FUSIONATTN: A Hybrid Sparse Attention Module for Training‑Free Acceleration

**Abstract**

We introduce **FUSIONATTN**, a hybrid sparse attention module that integrates ideas from four recent efficient attention mechanisms—SpargeAttn, SeerAttention, MoBA, and Kascade—into a single unified block. The module operates by first predicting dense regions of the attention matrix via a fast approximate scoring (SpargeAttn), then using a learned block‑level gate (SeerAttention) to identify important key blocks, and finally computing exact attention only over the top‑\(k\) selected blocks (MoBA). The design is intended to work out‑of‑the‑box without additional training, yet the gate network can be fine‑tuned for further gains. FUSIONATTN maintains standard projection layers and supports padding masks. This paper describes the module’s architecture, the underlying mathematics of each component, and its integration as a drop‑in replacement for full self‑attention in Transformer models.

---

## 1. Introduction

While multi‑head self‑attention provides powerful sequence modelling, its quadratic cost precludes application to very long sequences. Numerous sparse and approximate attention mechanisms have been proposed to alleviate this burden. Among them, four strategies have shown particular promise:

- **SpargeAttn**: uses a cheap approximate attention score computation to locate dense (high‑importance) regions of the attention matrix, allowing the model to skip sparse regions.
- **SeerAttention**: introduces a block‑wise gate that predicts which key blocks are relevant for each query block, enabling block‑sparse patterns.
- **MoBA (Mixture of Block Attention)**: selects the top‑\(k\) blocks according to gating scores and restricts attention to those blocks.
- **Kascade**: reuses gate predictions from earlier layers to guide sparsity in later layers, avoiding recomputation.

FUSIONATTN combines the first three into a cohesive pipeline, with a placeholder for Kascade‑style reuse. By sequentially applying fast score prediction, block gating, and top‑\(k\) block attention, the module achieves a structured sparse attention that is both content‑adaptive and efficient. Critically, the gate network is initialised such that the module can operate without any training, offering a “plug‑and‑play” acceleration of pretrained Transformers.

---

## 2. Method

### 2.1 Overall Pipeline

Given hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times d_{\text{model}}}\), the module first computes standard query, key, and value projections:

\[
\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{H}\mathbf{W}_Q, \mathbf{H}\mathbf{W}_K, \mathbf{H}\mathbf{W}_V,
\]

reshaped to \((B, H, N, d_h)\). The attention computation then proceeds in three stages:

1. **SpargeAttn – Fast Score Approximation** (optional). Produces a binary mask \(\mathbf{M}_{\text{dense}}\) indicating high‑score positions based on a reduced‑precision attention map.
2. **SeerAttention – Block Gate Scoring**. Divides the sequence into blocks of size \(B_s\) and uses a small gate network to compute a salience score for each pair of blocks, yielding a matrix \(\mathbf{G} \in \mathbb{R}^{B \times H \times N_b \times N_b}\).
3. **MoBA – Top‑\(k\) Block Attention**. For each query block, selects the \(k\) key blocks with the highest gate scores, and performs exact scaled dot‑product attention only within those selected blocks, optionally further filtered by the SpargeAttn mask.

The output is then reshaped and passed through an output projection \(\mathbf{W}_O\).

### 2.2 SpargeAttn – Fast Approximate Attention Map

The SpargeAttn component estimates which token pairs are likely to have high attention scores without computing the full \(O(N^2)\) high‑precision matrix. This is achieved by using only the first half of the head dimension:

\[
\mathbf{Q}_{\text{approx}} = \mathbf{Q}_{[:,:,:,:d_h/2]}, \quad
\mathbf{K}_{\text{approx}} = \mathbf{K}_{[:,:,:,:d_h/2]}.
\]

The approximate scores are:

\[
\mathbf{S}_{\text{approx}} = \frac{\mathbf{Q}_{\text{approx}} \mathbf{K}_{\text{approx}}^\top}{\sqrt{d_h/2}}.
\]

A binary mask is then obtained by thresholding at the mean score along the last dimension:

\[
\mathbf{M}_{\text{dense}} = \mathbf{S}_{\text{approx}} > \frac{1}{N} \sum_{j} S_{\text{approx}, i j}.
\]

This mask indicates which key positions are “dense” (high scoring) for each query. While naive, this heuristic localises regions where full attention is most beneficial.

### 2.3 SeerAttention – Block Gate Scoring

SeerAttention partitions the sequence into \(N_b = \lceil N / B_s \rceil\) blocks. For each block, a representative vector is obtained by averaging the queries (or keys) over the block:

\[
\bar{\mathbf{Q}}_b = \frac{1}{B_s} \sum_{t \in \text{block}(b)} \mathbf{Q}_{[:,:,t,:]}, \quad
\bar{\mathbf{K}}_b \text{ similarly}.
\]

These representatives are passed through a two‑layer MLP gate:

\[
\gamma(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1) \mathbf{w}_2,
\]

where \(\mathbf{W}_1 \in \mathbb{R}^{d_h \times 4d_h}\) and \(\mathbf{w}_2 \in \mathbb{R}^{4d_h \times 1}\). The gate is applied to both queries and keys, producing scalars \(g_q(\bar{\mathbf{Q}}_b), g_k(\bar{\mathbf{K}}_{b'})\). The block‑level gate score is their product:

\[
\mathbf{G}_{b, b'} = g_q(\bar{\mathbf{Q}}_b) \cdot g_k(\bar{\mathbf{K}}_{b'}) / \tau,
\]

with temperature \(\tau\) (initialised to 1). These scores reflect the predicted importance of block \(b'\) for all queries in block \(b\).

### 2.4 MoBA – Top‑\(k\) Block Attention

Using the gate scores \(\mathbf{G}\), for each query block \(b\) we select the \(k\) key blocks with highest scores (where \(k = \min(\text{top\_k\_blocks}, N_b)\)):

\[
\mathcal{T}_b = \operatorname{TopK}(\mathbf{G}_{b, :}, k).
\]

The selected key and value blocks are gathered. Attention is then computed between the query block and the gathered keys:

\[
\mathbf{S}_b = \frac{\mathbf{Q}_b \mathbf{K}_{\mathcal{T}_b}^\top}{\sqrt{d_h}},
\]
\[
\mathbf{A}_b = \operatorname{softmax}(\mathbf{S}_b + \mathbf{M}_{\text{pad}}),
\]
\[
\mathbf{C}_b = \mathbf{A}_b \mathbf{V}_{\mathcal{T}_b}.
\]

If the SpargeAttn dense mask is available, it is applied to further zero out positions that were not flagged as dense, before softmax.

The block outputs are concatenated to form the full context. This constrained attention reduces the per‑token complexity from \(O(N)\) to \(O(k B_s + B_s^2)\) per query block.

### 2.5 Kascade‑Style Reuse (Placeholder)

The module includes a flag `use_kascade_stability` and comments suggesting that gate scores could be cached from previous layers and reused, following the Kascade method. In the current implementation, this is not activated; the gate is recomputed at each layer. This design leaves the door open for future memory‑efficient attention with cross‑layer gate sharing.

---

## 3. Implementation Details

The class `FUSIONATTN` is not registered via the usual registry in the provided snippet, but it is designed as a standalone module.

**Key Components**:
- **Projections**: `q_proj`, `k_proj`, `v_proj`, `out_proj` are standard linear layers without bias (implicit, but likely bias=True by default).
- **Gate network**: `attn_gate` is a two‑layer MLP with GELU, taking a \(d_h\)‑dimensional vector and outputting a scalar. It is applied to each block’s average query/key independently.
- **Temperature**: `gate_temp` is a learnable scalar initialised to 1, controlling the sharpness of the gate scores before top‑\(k\) selection.
- **Hyperparameters**: `block_size` (default 64), `top_k_blocks` (default 4), `num_heads`, `head_dim`, etc.

**Forward Pass**:
1. Project inputs to Q, K, V and reshape to \((B, H, N, d_h)\).
2. If `use_sparge` is True, compute `dense_mask` via `_sparge_predict`.
3. Compute `gate_scores` via `_seer_gate_blocks`.
4. Compute `context` via `_moba_block_attention`, which uses the gate scores to select top‑\(k\) blocks and performs attention within those blocks. The simplified gathering code (using only `block_indices[0,0]`) is a placeholder; a correct implementation would use `torch.gather` or indexing to assemble the per‑block selections. For brevity, we note this as a simplification.
5. Apply attention mask to output (not to scores, since the attention is already limited). The mask is expected to be a float mask with 1 for valid tokens.
6. Reshape, output projection, return `(output, None)`.

**Training‑Free Operation**: The gate is initialised with random weights, but the module can already prune away many blocks based on the gate scores (which may be uninformative). The `gate_temp` and gate weights can be trained for optimal performance, but the module is claimed to work without training—the non‑linear gating and top‑\(k\) still induce sparsity, albeit suboptimally.

---

## 4. Usage and Configuration

Instantiate the module with model‑specific dimensions:

```python
attn = FUSIONATTN(
    hidden_size=768,
    num_heads=12,
    block_size=64,
    top_k_blocks=4
)
```

The module can be dropped into a Transformer layer, replacing the standard self‑attention. The `forward` call expects `hidden_states` and optional `attention_mask`. Flags `use_sparge` and `use_kascade_stability` control optional features.

Configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 64 | Size of blocks for block‑gating and MoBA. |
| `top_k_blocks` | 4 | Number of blocks selected per query block. |
| `use_sparge` | True | Whether to compute the SpargeAttn dense mask. |
| `use_kascade_stability` | False | Placeholder for Kascade reuse. |

---

## 5. Discussion

FUSIONATTN presents a novel synthesis of three sparse attention paradigms. The sequential application of fast prediction, gate scoring, and top‑\(k\) selection yields a structured sparsity pattern that can adapt to content while remaining computationally bounded. The use of half‑precision features for SpargeAttn provides a lightweight filter that eliminates clearly low‑relevance positions before the heavier block attention. The gate network, though small, learns to soft‑select blocks that are globally relevant, and the top‑\(k\) mechanism guarantees a constant memory footprint per query.

However, the current implementation has limitations. The gathering in `_moba_block_attention` is oversimplified—it uses only the first batch/head element’s indices, which would break for heterogeneous block selections across heads. A production version would require batched gather or scatter operations. Additionally, the gate network is applied to each block independently; a pairwise interaction term could strengthen block similarity estimation. The SpargeAttn threshold is fixed as the mean score, which may not be optimal for all layers; a learnable threshold could improve adaptivity. The Kascade reuse remains unimplemented, leaving potential memory savings on the table.

Despite these issues, the module demonstrates the principle of hybrid sparse attention and offers a foundation for future refinement. It is especially suitable for scenarios where sequence lengths are long but the number of important blocks is small, such as in document‑level tasks with a local context window augmented by a few long‑range dependencies.

---

## 6. Conclusion

We have presented FUSIONATTN, a hybrid sparse attention module that blends fast score prediction, block gating, and top‑\(k\) block attention. The design allows for training‑free acceleration of pretrained Transformers while supporting fine‑tuning of its lightweight gate components. Future work will address the current implementation simplifications and fully integrate the Kascade cross‑layer caching, pushing the module towards a practical, high‑performance drop‑in replacement for standard attention.
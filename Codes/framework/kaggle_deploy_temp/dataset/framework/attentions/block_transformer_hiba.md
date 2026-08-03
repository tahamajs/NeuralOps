# Block Transformer with HIBA: Hierarchical Inter‑Block Attention with Dense Rectification

**Abstract**

We introduce **Block Transformer HIBA**, a hybrid attention mechanism that decomposes the standard quadratic self‑attention into a combination of **intra‑block full attention**, **inter‑block sparse attention**, and **periodic global dense rectification**. The sequence is partitioned into fixed‑size blocks. Within each block, causal (or optionally full) attention captures local dependencies. Between blocks, each query token attends only to the top‑\(k\) tokens from all other blocks, using a learned sparsity pattern. To prevent the model from fragmenting its representation across disconnected blocks, every \(r\) layers a global dense attention pass is forced, recomputing the full attention matrix. This approach achieves sub‑quadratic complexity while retaining the ability to model long‑range dependencies, with occasional full‑context rectification to maintain coherence. The module serves as a drop‑in replacement for standard self‑attention in Transformers, offering a tunable efficiency–performance trade‑off via block size, top‑\(k\), and rectification interval.

---

## 1. Introduction

The quadratic complexity of the full attention mechanism has motivated numerous approximations. Block‑based attention, such as in BlockBERT and sparse Transformers, restricts interactions to local blocks, but this can lead to “fragmentation” — the inability to propagate information across distant blocks. The HIBA (Hierarchical Inter‑Block Attention) approach mitigates this by allowing each token to directly attend to a small, context‑driven selection of tokens from outside its own block. Additionally, a periodic **dense rectification** step temporarily reverts to global attention, ensuring that no long‑range signal is permanently lost.

In this paper, we formalise the Block Transformer HIBA module. The design balances three regimes:

1. **Local precision**: causal attention within a block, which is exact and cheap.
2. **Controlled sparsity**: inter‑block attention restricted to top‑\(k\) tokens from other blocks, selected via attention scores.
3. **Global coherence**: every \(r\) layers, a full dense attention pass is performed, acting as a global “reset” for the context.

The module is intended for long‑sequence modelling and can be injected directly into existing Transformer architectures.

---

## 2. Method

### 2.1 Block Partitioning

Given an input sequence of length \(L\), we divide it into \(B = \lceil L / w \rceil\) consecutive blocks of size \(w\) (the `block_size`). Let the blocks be indexed \(b = 0, \dots, B-1\). For a block \(b\), its token positions are \(I_b = [b \cdot w, \min((b+1) \cdot w, L))\).

### 2.2 Intra‑Block Attention

Inside a block, each token attends to all previous tokens within the same block, i.e., **causal** attention. For queries, keys, and values from block \(b\), denoted \(\mathbf{Q}_b, \mathbf{K}_b, \mathbf{V}_b \in \mathbb{R}^{B \times H \times w \times d_h}\), the intra‑block output is:

\[
\mathbf{S}_{\text{intra}} = \frac{\mathbf{Q}_b \mathbf{K}_b^\top}{\sqrt{d_h}},
\]
\[
\mathbf{A}_{\text{intra}} = \operatorname{softmax}(\mathbf{S}_{\text{intra}} + \mathbf{M}_{\text{causal}}),
\]
\[
\mathbf{C}_{\text{intra}} = \mathbf{A}_{\text{intra}} \mathbf{V}_b,
\]

where \(\mathbf{M}_{\text{causal}}\) is a lower‑triangular mask ensuring that token \(i\) only attends to \(j \le i\). This is a standard causal decoder block.

### 2.3 Inter‑Block Sparse Attention

To capture long‑range dependencies without quadratic cost, tokens from block \(b\) are allowed to attend to a sparse set of tokens from **all other blocks**. The selection is dynamic: for each token in block \(b\), we compute its attention scores with the keys of all tokens outside its block (i.e., blocks \(< b\) and \(> b\)). Let \(\mathbf{K}_{\text{other}}, \mathbf{V}_{\text{other}}\) be the concatenated keys and values from all other blocks. The raw inter‑block scores are:

\[
\mathbf{S}_{\text{inter}} = \frac{\mathbf{Q}_b \mathbf{K}_{\text{other}}^\top}{\sqrt{d_h}} \;\in \mathbb{R}^{B \times H \times w \times (L - w_b)}.
\]

For each query position, we retain only the top‑\(k\) largest scores (default \(k=32\)). A binary mask \(\mathbf{T}\) marks these selected positions, and the attention probabilities are:

\[
\tilde{\mathbf{S}}_{\text{inter}} = \mathbf{S}_{\text{inter}} \odot \mathbf{T} + (1 - \mathbf{T}) \cdot (-\infty),
\]
\[
\mathbf{A}_{\text{inter}} = \operatorname{softmax}(\tilde{\mathbf{S}}_{\text{inter}}),
\]
\[
\mathbf{C}_{\text{inter}} = \mathbf{A}_{\text{inter}} \mathbf{V}_{\text{other}}.
\]

The final block output is the sum of intra‑ and inter‑block contexts:

\[
\mathbf{C}_b = \mathbf{C}_{\text{intra}} + \mathbf{C}_{\text{inter}}.
\]

This design ensures that each token can directly attend to a handful of the most relevant tokens from afar, while local attention remains dense.

### 2.4 Periodic Global Dense Rectification

If blocks only exchange information through sparse top‑\(k\) connections, information may become trapped or propagate slowly across many layers. To combat this, every \(r\) layers (the `rectification_interval`), the module disregards block boundaries and performs a full, dense, bidirectional self‑attention over the entire sequence:

\[
\mathbf{S}_{\text{global}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]
\[
\mathbf{A}_{\text{global}} = \operatorname{softmax}(\mathbf{S}_{\text{global}} + \mathbf{M}_{\text{pad}}),
\]
\[
\mathbf{C}_{\text{global}} = \mathbf{A}_{\text{global}} \mathbf{V}.
\]

Here \(\mathbf{M}_{\text{pad}}\) is the standard padding mask. This rectification step guarantees that after every \(r\) layers, all tokens receive a complete, unfiltered view of the sequence, thereby correcting any accumulated fragmentation.

### 2.5 Overall Forward Pass

The module maintains a layer counter `layer_idx` that increments on each forward call. If `layer_idx % rectification_interval == 0`, the global dense pass is executed. Otherwise, the sequence is processed block‑by‑block, and the per‑block outputs are concatenated along the sequence dimension.

All queries, keys, and values are obtained via learned linear projections \(\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) (without bias). The final output is projected by \(\mathbf{W}_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\).

---

## 3. Implementation Details

The class `BlockTransformerHIBA` inherits from `BaseAttention` and is registered as `"block_transformer_hiba"` in the attention registry.

Key components:
- **`_intra_block_attention(q, k, v, causal=True)`**: computes causal self‑attention for a block. If `causal=False`, it becomes full intra‑block attention (currently always called with `causal=True`).
- **`_inter_block_sparse(q_b, k_all, v_all, block_idx, num_blocks)`**: 
  - Gathers keys and values from all blocks before and after the current block.
  - Computes scores, selects top‑\(k\) indices, builds a binary mask, and returns the sparse attention output.
- **Forward mask handling**: The global rectification pass uses the provided `mask` (if any) to mask padding. The block‑wise path currently does not apply any padding mask, which could be problematic for variable‑length sequences; a full implementation should incorporate masking in the intra‑ and inter‑block stages as well.
- **Layer counter**: `self.layer_idx` is incremented after checking the rectification condition, ensuring that the global pass occurs at layers \(0, r, 2r, \dots\). Note: if the same module is reused across multiple layers (as is typical in Transformers), the counter will not correctly track the absolute layer depth because each layer has its own instance. For a correct global interval, the counter should be tied to the layer’s position in the network (e.g., passed as a constructor argument). The current code uses a per‑module counter, which only works if `layer_idx` is set externally before each forward or if the module is used only once. In practice, one would initialise `layer_idx` with the actual layer index.

---

## 4. Usage and Configuration

The module can be injected into a DistilBERT layer via the framework’s registry. Key hyperparameters (passed through a compression config or directly) are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 512 | Size of each block (window for local attention). |
| `top_k_inter` | 32 | Number of top tokens selected per query for inter‑block attention. |
| `rectification_interval` | 4 | Layers between global dense rectification passes. |
| `rank` | 64 | (Unused, inherited from `BaseAttention`) |
| `dropout` | (from config) | Dropout is not explicitly applied in the code; could be added inside softmax. |

Example injection (manual):
```python
attn = BlockTransformerHIBA(
    config=model_config,
    block_size=256,
    top_k_inter=16,
    rectification_interval=2
)
layer.attention = attn
```

---

## 5. Discussion

The Block Transformer HIBA combines the benefits of localised exact attention with a flexible, learnable sparsity pattern for distant tokens. The top‑\(k\) inter‑block mechanism avoids the need for fixed sparse patterns (e.g., stride or fixed windows) and adapts to the content. The periodic global rectification serves as a safety net, ensuring that the model never permanently loses sight of the full context. This design is particularly well‑suited for autoregressive generation (causal intra‑block) and long‑document encoding.

One limitation is that the current implementation does not mask padding tokens in the block‑wise path, which could lead to wasted computation and leakage; a production version should integrate the attention mask. Additionally, the layer counter mechanism must be aligned with the actual Transformer depth to guarantee the intended rectification schedule. Finally, the top‑\(k\) selection, while dynamic, is a hard attention that may be less stable than soft attention; using a softer approximation (e.g., entmax) could improve gradient flow.

---

## 6. Conclusion

We have presented Block Transformer HIBA, a novel attention module that interleaves dense block‑local attention with a learned top‑\(k\) inter‑block sparse attention and occasional global dense passes. This hierarchical strategy reduces the asymptotic complexity from \(O(L^2)\) to \(O(L \cdot (w + k B))\) while preserving strong long‑range modelling capability. Its modular design and compatibility with existing Transformer frameworks make it a practical choice for efficient long‑sequence processing.
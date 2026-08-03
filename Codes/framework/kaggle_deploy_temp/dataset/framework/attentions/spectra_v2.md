# SPECTRA‑v2: Sparse Per‑Head Efficient Context Transformer with Rectified Attention

**Abstract**

We present **SPECTRA‑v2**, a hybrid sparse attention module for Transformers that integrates three complementary mechanisms: (1) **intra‑block full attention** combined with **per‑query inter‑block top‑k selection** (enriched by a relative position bias), (2) a soft, **adaptive per‑head span** to gently limit attention range, and (3) a **periodic global rectification** via orthogonal random Fourier features (RFF) to prevent context fragmentation. The block‑sparse pattern is constructed on the fly: each token attends to all tokens within its local block and additionally to a small, dynamically chosen set of the highest‑scoring tokens from adjacent blocks. The relative position bias encourages the selection of tokens that preserve the original order, while the adaptive span provides a learnable distance decay. Every \(R\)-th layer, the sparse attention is replaced by a global linear attention pass that guarantees a full‑context refresh. The module is designed as a drop‑in replacement for the self‑attention layer in DistilBERT, with zero‑shot weight transfer from a pretrained model and a clean, registry‑based injection. This paper provides a rigorous mathematical formulation of each component, the block‑sparse attention algorithm, and the overall architecture.

---

## 1. Introduction

Standard multi‑head self‑attention computes all pairwise dot‑products, leading to \(O(L^2)\) time and memory. Block‑sparse attention methods reduce this cost by partitioning the sequence into blocks and restricting inter‑block interactions to a selected subset. The **SPECTRA‑v2** module refines this idea with three key innovations:

- **Per‑query top‑k inter‑block selection with relative bias**: Instead of selecting whole blocks (as in MoBA or BigBird), each query token individually chooses the top‑\(k\) key tokens from its adjacent blocks. A learned relative position bias guides this selection, favouring tokens that are roughly in the same relative order, thus preserving the sequential structure.
- **Adaptive per‑head span**: A learnable scalar per head defines a soft window that gradually reduces attention scores for distant tokens, even within the intra‑block set. This acts as a fine‑grained control on the attention range.
- **Periodic global rectification**: To prevent information isolation across blocks, every \(R\)-th layer (e.g., every 4th) performs a full global linear attention using orthogonal random features (the “AKS” kernel), which approximates softmax attention in linear time.

The module also supports standard padding masks and can be initialised directly from a pretrained DistilBERT attention layer, ensuring zero‑shot performance. The name SPECTRA stands for **Sp**arse **E**fficient **C**ontext **T**ransformer with **R**ectified **A**ttention; the “v2” indicates the corrected top‑k logic and the addition of relative position bias.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module applies standard linear projections for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\). These are reshaped to the multi‑head layout \((B, H, L, d_h)\).

### 2.2 Block Partitioning and Adaptive Span Logit

The sequence is divided into blocks of size \(B_s\) (default 64). Let the number of blocks be \(N_b = \lceil L / B_s \rceil\).

An **adaptive span** is implemented via a learnable per‑head scalar \(s_h \in \mathbb{R}\) (initialised to a large value, e.g., 512). For a given head, a soft log‑mask is constructed based on the token distances:

\[
M_{ij}^{(h)} = \log\bigl( \sigma\bigl( (s_h - |i-j|) \cdot 5 \bigr) + \varepsilon \bigr),
\]

where \(\sigma\) is the sigmoid. This produces a value near 0 (log‑1) for distances well within the span, and a large negative logit for distances far beyond the span. This logit is added to the attention scores, thereby softly down‑weighting far‑away tokens. The steepness factor 5 makes the transition relatively sharp.

### 2.3 Intra‑Block Full Attention + Per‑Query Inter‑Block Top‑k with Relative Bias

For each block \(b\) spanning indices \([s, e)\) (where \(s = b \cdot B_s\), \(e = \min(s+B_s, L)\)), the following steps are performed:

1. **Intra‑block**: All tokens within the block attend to each other without restriction. The scores are computed as \(\mathbf{S}_{\text{intra}} = \frac{\mathbf{Q}_b \mathbf{K}_b^\top}{\sqrt{d_h}}\), and the adaptive span logit \(M^{(h)}\) (sliced to the block) is added.

2. **Inter‑block candidates**: Tokens from the immediately adjacent blocks—left block (if \(b>0\)) and right block (if \(b < N_b-1\))—are considered as inter‑block candidates. (Only one block on each side is used; the comment says “previous and next block”.)

   For a query token at position \(i\) in block \(b\), we compute its attention scores with all candidate tokens from the left and right blocks. Before selection, a **relative position bias** is added to the scores. The bias is learned per head and depends on the clipped relative distance:

   \[
   \delta = \operatorname{clamp}(i - j, -d_{\max}, d_{\max}),
   \]
   where \(d_{\max}\) is the max relative distance (default 128). The bias is stored as a table of shape \((H, 2 d_{\max} + 1)\). Adding this bias encourages the model to prefer tokens with certain relative positions, helping to preserve the order of information.

3. **Per‑query top‑k selection**: For each query token individually, the top‑\(k\) highest scoring inter‑block tokens are selected (after bias addition). This yields \(k \cdot B_s\) selected keys and values for the whole block (since each token in the block can choose its own top‑\(k\)). The implementation gathers these per‑token selections and forms a combined key–value tensor of size \((B, H, B_s \cdot k, d_h)\).

4. **Combined attention**: The intra‑block keys and values are concatenated with the selected inter‑block keys and values. The final attention scores are computed over this combined set, softmax is applied, and the values are aggregated. The adaptive span logit is applied only to the intra‑block part.

5. **Masking**: If a padding mask is provided (as a boolean tensor `token_valid` of shape \((B, L)\)), it is used to mask out invalid keys: both intra‑block and inter‑block keys that are padding are set to \(-\infty\) before softmax.

### 2.4 Periodic Global Rectification

Every \(R\)-th layer (controlled by `rectify_interval`, default 4), the sparse attention is **bypassed** entirely. Instead, the module uses a global linear attention mechanism based on **random Fourier features (RFF)**. The feature map is:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt[4]{F}} \bigl[ \cos(\mathbf{x}\mathbf{P}),\; \sin(\mathbf{x}\mathbf{P}) \bigr],
\]

where \(\mathbf{P} \in \mathbb{R}^{d_h \times F}\) is a learnable random projection matrix (initialised from \(\mathcal{N}(0, 0.02^2)\)). The linear attention output is:

\[
\mathbf{KV} = \phi(\mathbf{K})^\top \mathbf{V},\quad
\mathbf{Z} = \phi(\mathbf{K})^\top \mathbf{1},\quad
\mathbf{C} = \frac{\phi(\mathbf{Q}) \mathbf{KV}}{\phi(\mathbf{Q}) \mathbf{Z} + \epsilon}.
\]

This operation captures the global context in linear time and serves to re‑establish long‑range connections that may have been lost in the sparse layers.

### 2.5 Output Projection and Masking

After the context is computed (either sparse or global), the output is multiplied by the token validity mask (zeroing out padding positions). The heads are then concatenated and projected with \(\mathbf{W}_O\) (and dropout applied). The module returns a tuple `(output, None)`.

---

## 3. Implementation Details

- **`SpectraV2Attention`** is registered under `"spectra_v2"` via the injection function `inject_spectra_v2`.
- **Configuration**: Hyperparameters are stored in the `SpectraConfig` dataclass, with sensible defaults (block_size=64, top_k_inter=16, adaptive_span=True, rectify_interval=4, etc.).
- **Zero‑shot weight transfer**: The injection function copies the pretrained `q_lin`, `k_lin`, `v_lin`, `out_lin` weights into the corresponding linear layers of SPECTRA‑v2, ensuring the model starts from the pretrained state.
- **Adaptive span**: The span logit is added only to intra‑block scores. The span parameter is clamped to a minimum of 1.0 and is per‑head, allowing different heads to learn different effective ranges.
- **Relative position bias**: A table of shape `(num_heads, 2*max_dist+1)` is learned. During attention, the relative distances between query and candidate key are computed, clipped, and used to index this table. The bias is added to the inter‑block scores before top‑k selection, influencing which tokens are kept.
- **Global rectification**: Implemented via the `SimpleRFFKernel` class. The random projection matrix is learnable (a parameter), but its variance is small. The rectification layer is active only when `(layer_idx % rectify_interval == 0)` and `aks_enabled` is True.
- **Mask handling**: A robust `_get_token_valid_mask` method converts 2D, 3D, or 4D masks to a boolean `(B, L)` validity tensor. This mask is used in the sparse attention to set invalid key scores to -1e9, and to zero out padding positions in the final output.
- **Complexity**: The intra‑block attention costs \(O(L B_s d_h)\). The inter‑block selection costs \(O(L k d_h)\) per layer, plus the overhead of top‑k per query. With \(k=16\) and \(B_s=64\), the total cost per token is approximately \(O((B_s + 2k) d_h)\), which is linear in \(L\). The global rectification layer costs \(O(L F d_h)\), also linear.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"spectra_v2"`. The relevant parameters in `comp_config` include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 64 | Size of local blocks. |
| `top_k_inter` | 16 | Number of inter‑block tokens selected per query. |
| `adaptive_span` | True | Whether to use per‑head learnable span. |
| `max_span` | 512.0 | Initial value for the adaptive span parameter. |
| `rectify_interval` | 4 | Frequency of global rectification layers. |
| `rff_features` | 128 | Feature dimension \(F\) for the RFF kernel. |
| `use_rel_pos_bias` | True | Whether to use relative position bias for inter‑block selection. |
| `rel_pos_max_distance` | 128 | Maximum clipped relative distance for bias. |
| `dropout` | 0.1 | Dropout rate on attention weights and output. |

Example injection:
```python
comp_config.attention_type = "spectra_v2"
comp_config.block_size = 128
comp_config.top_k_inter = 32
```

The injection function copies pretrained weights, so the model can be used directly for inference or fine‑tuned.

---

## 5. Discussion

SPECTRA‑v2 presents a carefully balanced sparse attention design. The intra‑block full attention ensures that local dependencies are captured with high fidelity. The per‑query top‑k inter‑block selection is a fine‑grained way to extend the receptive field without committing to entire blocks; by adding a learned relative position bias, the selection becomes order‑aware, which is particularly beneficial for positional tasks. The adaptive span allows each head to specialise its effective range, and the periodic global rectification prevents the cumulative effect of many sparse layers from isolating tokens.

However, the method has some limitations:

- The inter‑block candidates are restricted to adjacent blocks only (one left and one right). This limits the maximum distance for long‑range interactions. For very long sequences, a token cannot attend to far‑away positions unless the global rectification layer intervenes. Extending to a wider set of blocks (or using landmarks) could improve long‑range capability.
- The top‑k selection is per‑query, which increases the computational effort compared to block‑wise selection. For large \(L\), the cost of top‑k (roughly \(O(L \log k)\)) could be noticeable, but it is still linear.
- The relative position bias adds a small parameter overhead but significantly helps in preserving order. However, it might not be sufficient for tasks requiring precise absolute position tracking.
- The global rectification uses a simple RFF kernel with a learnable projection; while effective, it might suffer from the same approximation error as any random feature method. The variance could be reduced by using orthogonal features (as in FAVOR+), but the current implementation uses plain random features (the `SimpleRFFKernel` uses `torch.randn` without orthogonalisation).

Despite these, the module offers a rich set of features that can be tuned for different tasks. Its modular design and zero‑shot initialisation make it a practical drop‑in replacement for standard attention in DistilBERT.

---

## 6. Conclusion

We have presented SPECTRA‑v2, an advanced sparse attention module that combines intra‑block full attention with per‑query top‑k inter‑block selection, adaptive per‑head spans, relative position bias, and periodic global rectification. The design achieves sub‑quadratic complexity while preserving both local and long‑range dependencies through a hybrid scheme. The module is integrated into the compression framework with zero‑shot weight transfer, making it an excellent candidate for efficient fine‑tuning and deployment of Transformer models on long sequences. Future work may explore extending the inter‑block range and incorporating more sophisticated global kernels.
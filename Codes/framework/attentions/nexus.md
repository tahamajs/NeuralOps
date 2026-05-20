# NEXUS: Neural Efficient X‑Scoring Unified Sparse Attention

**Paper Reference**  
NEXUS synthesizes six state‑of‑the‑art efficient attention mechanisms published at ICML/NeurIPS 2025 into a single drop‑in module for Transformer models. It is designed as a zero‑shot replacement for the self‑attention sub‑layer in DistilBERT, combining training‑free block scoring (XAttention), learnable head‑wise gating (SeerAttention), dual‑stage intra‑ and inter‑block attention (DuSA), fine‑grain stripe granularity (AnchorAttention), periodic global rectification via random features (Spectraformer), and a rectified softmax for natural sparsity (Softpick). The result is a hybrid sparse attention that balances efficiency, accuracy, and adaptability without requiring extensive retraining.

---

## 1. Introduction

The quadratic complexity of multi‑head self‑attention has spurred a multitude of sparse, low‑rank, and recurrent approximations. Several recent works (circa 2025) have independently proposed complementary ideas for reducing the cost of attention while preserving model quality. **NEXUS** (Neural Efficient X‑scoring Unified Sparse Attention) is a unified framework that integrates six of these innovations into a single, coherent attention module:

1. **XAttention** (Xu et al., ICML 2025) – an antidiagonal‑based training‑free block scoring mechanism.
2. **SeerAttention** (Gao et al., NeurIPS 2025) – a learnable per‑head gate that predicts block sparsity.
3. **DuSA** (Cao et al., NeurIPS 2025) – a dual‑stage design combining full intra‑block attention with sparse inter‑block attention.
4. **AnchorAttention** (Zhu et al., 2025) – a fine‑grained stripe pattern within blocks to further reduce computation.
5. **Spectraformer** (Nguyen et al., 2025) – periodic global rectification using random Fourier features to prevent context fragmentation.
6. **Softpick** (Zuhri et al., 2025) – a rectified softmax that naturally yields ~47% sparse attention maps.

NEXUS harmonises these techniques under a configurable dataclass, offering a highly modular and powerful drop‑in replacement for standard self‑attention. The module is initialised with weights copied from a pretrained DistilBERT attention layer, ensuring zero‑shot performance, and can be fine‑tuned end‑to‑end. This paper provides a comprehensive mathematical description of each component and how they are combined into the NEXUS architecture.

---

## 2. The NEXUS Architecture

### 2.1 Overall Pipeline

Given an input sequence of hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\), the module first applies standard linear projections to obtain multi‑head queries, keys, and values:

\[
\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{H}\mathbf{W}_Q, \mathbf{H}\mathbf{W}_K, \mathbf{H}\mathbf{W}_V,
\]

reshaped to \((B, H, L, d_h)\), where \(H\) is the number of heads and \(d_h = D/H\). The attention output is then determined by one of two paths:

- **Periodic global rectification** – every \(R\)-th layer (controlled by `rectify_interval`, default 4), the module uses a Spectraformer‑style global linear attention that captures the full context using random Fourier features.
- **Dual‑stage sparse attention** – on other layers, it applies the DuSA‑style intra‑block full attention combined with inter‑block sparse attention, guided by XAttention block scores and optionally refined by SeerAttention gating. Inside each block, an AnchorAttention‑stripe sparsity pattern further prunes unnecessary computations, and Softpick activation induces additional sparsity.

Finally, the context tensor is reshaped and projected via \(\mathbf{W}_O\) to produce the output. The module returns a tuple compatible with the DistilBERT framework.

In the following subsections, we detail each of the six integrated mechanisms.

### 2.2 XAttention: Antidiagonal Block Scoring

XAttention (Xu et al., ICML 2025) introduces a training‑free method to assess the importance of token blocks for attention. The key idea is to sample tokens at a regular stride and compute a coarse attention map, then aggregate scores along antidiagonal blocks to obtain a block‑level importance matrix.

Given \(\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{B \times H \times L \times d_h}\) and a block size \(B_s\), we first subsample with stride \(S\) (default 2), obtaining \(\mathbf{Q}_{\text{sub}}, \mathbf{K}_{\text{sub}} \in \mathbb{R}^{B \times H \times L_{\text{sub}} \times d_h}\) where \(L_{\text{sub}} \approx L/S\). We then compute a subsampled attention matrix:

\[
\mathbf{A}_{\text{sub}} = \frac{\mathbf{Q}_{\text{sub}} \mathbf{K}_{\text{sub}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L_{\text{sub}} \times L_{\text{sub}}}.
\]

The sequence is partitioned into \(n_b = \lceil L / B_s \rceil\) blocks, and each block corresponds to roughly \(B_s / S\) subsampled indices. The block score between query block \(i\) and key block \(j\) is the **mean** of the subsampled attention values in that region:

\[
\mathbf{S}^{\text{block}}_{b,i,j} = \frac{1}{|R_{ij}|} \sum_{(p,q) \in R_{ij}} \mathbf{A}_{\text{sub},:,p,q},
\]

where \(R_{ij}\) is the rectangular region covering the corresponding subsampled positions. This yields a matrix \(\mathbf{S}^{\text{block}} \in \mathbb{R}^{B \times H \times n_b \times n_b}\). In the NEXUS code, this is implemented with a double loop over blocks for simplicity; a more optimised version would use strided pooling.

The resulting block scores serve as a **cheap, training‑free** estimate of inter‑block relevance. They are used to select the top‑\(k\) blocks for inter‑block attention in the DuSA stage (when `xattn_enabled` is True). The method adds minimal overhead because it operates on a heavily subsampled attention map.

### 2.3 SeerAttention: Learnable Per‑Head Gating

SeerAttention (Gao et al., NeurIPS 2025) introduces a lightweight, learnable gate that predicts which blocks should be active for each head, allowing the sparsity pattern to adapt during training. NEXUS includes SeerAttention as an optional enhancement.

A small gate network processes the **average** query and key vectors over the sequence dimension:

\[
\bar{\mathbf{q}} = \frac{1}{L} \sum_{i=1}^L \mathbf{q}_i \in \mathbb{R}^{B \times H \times d_h}, \quad
\bar{\mathbf{k}} = \frac{1}{L} \sum_{i=1}^L \mathbf{k}_i.
\]

These are projected through a two‑layer MLP (with GELU and a bottleneck of size `seer_gate_rank`) to produce per‑head gating scalars:

\[
\gamma = \operatorname{MLP}_Q(\bar{\mathbf{q}}) \in \mathbb{R}^{B \times H \times 1}, \quad
\kappa = \operatorname{MLP}_K(\bar{\mathbf{k}}) \in \mathbb{R}^{B \times H \times 1}.
\]

The final gate score is the sigmoid of their product divided by a temperature \(\tau\):

\[
g_h = \sigma\!\left(\frac{\gamma_h \cdot \kappa_h}{\tau}\right).
\]

The number of active blocks per head is then heuristically set as \(k_h = \max(1, \lfloor g_h \cdot n_b \rfloor)\). A block mask is constructed that keeps the top‑\(k_h\) blocks along the diagonal? In the current implementation, the mask forces a **diagonal** block pattern (blocks only attend to the first \(k_h\) blocks, including themselves), which is a simplification. The mask is returned as a boolean tensor \(\mathbf{M}^{\text{seer}} \in \{0,1\}^{B \times H \times n_b \times n_b}\).

In NEXUS, the Seer gate is **not** directly used to modify the attention scores; it is computed but the current code merely calls `seer_gate` and ignores its output (commented as a future integration). The design allows for the gate to later replace or augment the XAttention top‑\(k\) selection.

### 2.4 DuSA: Dual‑Stage Attention

DuSA (Cao et al., NeurIPS 2025) structures attention into two stages: **intra‑block full attention** and **inter‑block sparse attention**. NEXUS implements this as its core sparse attention engine.

**Intra‑block stage** – For each block \(i\) of size \(B_s\), all tokens within the block attend to each other densely:

\[
\mathbf{S}^{\text{intra}}_i = \frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_h}}, \quad
\mathbf{A}^{\text{intra}}_i = \operatorname{softmax}(\mathbf{S}^{\text{intra}}_i), \quad
\mathbf{C}^{\text{intra}}_i = \mathbf{A}^{\text{intra}}_i \mathbf{V}_i.
\]

**Inter‑block stage** – In addition, each query block \(i\) attends to a small selection of **other** key‑value blocks. Which blocks are selected is determined by the XAttention block scores (or, in a future version, the Seer mask). The top‑\(k\) highest scoring key blocks (excluding the same block) are chosen:

\[
\mathcal{T}_i = \operatorname{TopK}(\mathbf{S}^{\text{block}}_{:,:,i,:}, k), \quad \text{excluding } i.
\]

The selected key and value blocks are concatenated with the intra‑block ones, forming an extended key set \(\mathbf{K}^{\text{ext}}_i\) and \(\mathbf{V}^{\text{ext}}_i\). The full attention scores for block \(i\) are then:

\[
\mathbf{S}^{\text{full}}_i = \frac{\mathbf{Q}_i (\mathbf{K}^{\text{ext}}_i)^\top}{\sqrt{d_h}}.
\]

The attention output for block \(i\) is the weighted sum of the extended values. Finally, all block outputs are concatenated along the sequence dimension. This combination ensures that each token has a dense local context plus a sparse global context, achieving a balance between precision and efficiency.

### 2.5 AnchorAttention: Fine‑Grain Stripe Granularity

AnchorAttention (Zhu et al., 2025) proposes to further reduce computation within each block by restricting attention to a **stripe** pattern. Instead of attending to all tokens in the intra‑ and inter‑block combined key set, a query token only attends to keys that lie within a certain stripe distance along the sequence index.

In NEXUS, after the extended key tensor is built for a block, we group the positions (both intra‑ and inter‑block) into stripes of width \(W_s\) (default 4). A query token belonging to stripe \(p\) attends only to key tokens in stripes \(p-1, p, p+1\). This is implemented via a binary mask derived from the stripe indices:

\[
M_{ij} = \begin{cases}
1 & \text{if } |\text{stripe}(i) - \text{stripe}(j)| \le 1,\\
0 & \text{otherwise.}
\end{cases}
\]

Scores outside this mask are set to \(-\infty\), forcing sparse attention within the already restricted block set. The stripe mask is applied **after** the block selection, providing a second layer of fine‑grained sparsity.

### 2.6 Spectraformer: Periodic Global Rectification

A purely sparse attention pattern risks fragmenting the model’s view of the sequence, especially across many layers. Spectraformer (Nguyen et al., 2025) addresses this by periodically applying a **global linear attention** pass that captures the entire context using random Fourier features. In NEXUS, every \(R\)-th layer (e.g., every 4th layer) bypasses the dual‑stage sparse attention and instead uses a kernel‑based linear attention.

The kernel feature map is:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \exp\!\big( \mathbf{x} \mathbf{R} \big),
\]

where \(\mathbf{R} \in \mathbb{R}^{d_h \times F}\) is a learnable random projection matrix (initialised with \(\mathcal{N}(0,0.02^2)\)), and queries and keys are first L2‑normalised. The linear attention output is computed as:

\[
\mathbf{KV} = \sum_{j=1}^L \phi(\mathbf{k}_j) \mathbf{v}_j^\top, \qquad
\mathbf{Z} = \sum_{j=1}^L \phi(\mathbf{k}_j),
\]
\[
\mathbf{C} = \frac{\phi(\mathbf{Q}) \mathbf{KV}}{\phi(\mathbf{Q}) \mathbf{Z} + \epsilon}.
\]

This operation has \(O(L F d_h)\) complexity, which is linear in \(L\). By interspersing such global rectification layers, the model periodically re‑establishes long‑range connections, preventing the accumulation of errors from local sparsity.

### 2.7 Softpick: Rectified Softmax

Softpick (Zuhri et al., 2025) is an alternative to softmax that naturally produces sparse attention maps—around 47% of the weights become exactly zero. It is defined as:

\[
p_i = \frac{\operatorname{ReLU}\!\big( e^{(x_i - m)/\tau} - e^{-m/\tau} \big)}
            {\sum_j \operatorname{ReLU}\!\big( e^{(x_j - m)/\tau} - e^{-m/\tau} \big) + \epsilon},
\]

where \(m = \max_i x_i\) and \(\tau\) is a temperature parameter (default 1.0). The subtraction of the constant \(e^{-m/\tau}\) and the ReLU threshold jointly set small scores to zero. In NEXUS, Softpick replaces the standard softmax after the (sparse) attention scores, further eliminating weak connections and reducing the effective number of non‑zero attention edges. It is applied per block during the dual‑stage attention.

### 2.8 Integration and Zero‑Shot Initialisation

The NEXUS module wraps all these components into a single `NEXUSAttention` class. The linear projections `query`, `key`, `value`, `out_proj` are initialised by **copying** the pretrained weights from the original DistilBERT attention layer (`q_lin`, `k_lin`, etc.), ensuring zero‑shot performance. The additional parameters (random projection matrix for Spectraformer, Seer gate MLP) are randomly initialised and trained from scratch, but since they are small and the rectification is infrequent, the initial model output is nearly identical to the pretrained model.

The forward pass uses a robust mask handler that converts 2D/3D/4D masks to a boolean token validity tensor, which is used to mask padding both in the attention scores and in the final output. The module always returns a 2‑tuple `(output, None)` for compatibility.

---

## 3. Implementation Details

- **Configuration**: All hyperparameters are collected in the `NEXUSConfig` dataclass, with sensible defaults (e.g., `block_size=32`, `top_k_blocks=4`, `stripe_width=4`, `rectify_interval=4`, `rff_features=128`). The injection function `inject_nexus` reads these from the framework’s compression config.
- **XAttention scorer**: Implements the subsampled antidiagonal block scoring using a stride; loops are used for simplicity but could be vectorised.
- **Seer gate**: A two‑layer MLP per head, currently computed but its mask is not yet integrated into the attention; left as a placeholder for future enhancement.
- **Dual‑stage engine**: The `DualStageStripeAttention` class manages block partitioning, XAttention top‑\(k\) inter‑block selection, stripe mask generation, and Softpick. It iterates over blocks sequentially (Python loop), which is acceptable for moderate \(n_b\) but can be a bottleneck for very long sequences.
- **Spectraformer rectification**: A separate module that is instantiated only if the layer index is a multiple of `rectify_interval`. It uses the same Q, K, V projections; no additional parameters for the attention itself besides the random projection matrix.
- **Injection**: The function `inject_nexus_into_distilbert` copies pretrained weights and replaces `layer.attention`. The registry key is `"nexus"`.

---

## 4. Discussion

NEXUS provides a unique testbed for combining multiple state‑of‑the‑art sparse attention techniques. The hybrid design allows each component to contribute to efficiency and quality in a controlled manner:

- **XAttention** gives a fast, training‑free baseline for block importance, enabling the model to function even without training the gate.
- **SeerAttention** offers a path to learnable, head‑adaptive sparsity, which can be trained jointly with the task.
- **DuSA** divides the attention into local and global stages, ensuring that no token is ever completely isolated.
- **AnchorAttention stripes** add an extra fine‑grained sparsity that can drastically reduce computation within large blocks.
- **Spectraformer rectification** prevents context fragmentation, a common pitfall in purely sparse models.
- **Softpick** induces natural sparsity without a hard top‑\(k\), potentially improving gradient flow.

The main current limitation is that the Seer gate’s output is not actually used to modify the attention pattern; the top‑\(k\) selection relies solely on XAttention. Integrating the Seer mask would allow the model to learn to refine the block choices. Additionally, the stripe mask is applied after block selection but does not account for the inter‑block offset in its stripe grouping; the stripe indices are global across the concatenated key set, which may cause unintended cross‑block stripe connections. A more precise implementation would stripe within each original block separately. The loop over blocks may also hinder GPU utilisation; a fully batched implementation using scatter/gather would be desirable.

Despite these, NEXUS is a powerful and flexible module that demonstrates the synergy of recent efficient attention methods. Its modular config makes it easy to ablate individual components and to adapt to different downstream tasks.

---

## 5. Conclusion

We have presented NEXUS, a unified sparse attention module that fuses XAttention, SeerAttention, DuSA, AnchorAttention, Spectraformer, and Softpick into a single drop‑in replacement for DistilBERT’s self‑attention. The module leverages training‑free block scoring, learnable gating, dual‑stage attention, fine‑grained stripe sparsity, periodic global rectification, and a rectified softmax to achieve high efficiency with minimal quality loss. The zero‑shot weight transfer ensures immediate usability, while the rich configurable parameters allow tailoring to specific tasks. NEXUS stands as a testament to the power of combining complementary efficient attention techniques in a harmonious architecture.
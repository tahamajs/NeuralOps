# PHANTOM: Parameter‑Hashed AttentioN with Token‑wise Orthogonal Mapping

**Abstract**

We present **PHANTOM**, a dual‑compression attention module for Transformers that couples a **CUR matrix decomposition** of the projection weights with a **locality‑sensitive hashing (LSH)** based sparse attention pattern, both wrapped with a **softpick rectified softmax** to induce natural sparsity. The CUR factorisation approximates each pretrained weight matrix \( \mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \) via a product \( \mathbf{C} \mathbf{U} \mathbf{R} \), where \( \mathbf{C} \) and \( \mathbf{R} \) are column and row subsets selected according to leverage scores, and \( \mathbf{U} \) is a small linking matrix. This yields a low‑rank compressed representation stored in two linear layers, enabling zero‑shot initialisation from a pretrained model. The attention itself is performed not by a full dot‑product matrix but by multi‑round angular LSH: queries and keys are normalised and hashed via random hyperplane projections, and tokens only attend to others in the same hash bucket. This reduces the expected number of pairwise comparisons to roughly \( O(N \cdot B_{\text{avg}}) \), where \( B_{\text{avg}} \) is the average bucket size. Within each bucket, attention weights are computed using the softpick function, which further zeroes out weak connections. The module is designed as a drop‑in replacement for the self‑attention layer in DistilBERT, integrating with the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of the CUR factorisation, the LSH attention mechanism, and the overall architecture, discussing both the theoretical foundations and the practical implementation choices.

---

## 1. Introduction

The computational and memory footprint of self‑attention in Transformers motivates a wide range of efficiency methods. Two popular strategies are **low‑rank weight compression** and **sparse attention patterns**. The **PHANTOM** module combines both: it replaces each dense projection matrix with a **CUR decomposition** that retains the most important rows and columns of the original weight, and it replaces the all‑to‑all attention with a **locality‑sensitive hashing (LSH)** scheme that only computes dot‑products between tokens that fall into the same hash bucket. Additionally, a **softpick** activation replaces the standard softmax, further enhancing sparsity and potentially improving gradient flow.

The CUR decomposition is chosen over a standard SVD because the factor matrices \( \mathbf{C} \) and \( \mathbf{R} \) are actual columns and rows of the original weight, which can be more interpretable and sometimes yield better approximations for certain matrices. Leverage scores derived from the singular vectors guide the selection, ensuring that the most influential directions are preserved. The result is a pair of linear layers—a down‑projection to rank \( r \) and an up‑projection back—that drastically reduce the number of trainable parameters while initialising from pretrained weights for zero‑shot performance.

The LSH attention uses an **angular hash** (random hyperplane projections) to group queries and keys into buckets. This approach is inspired by the Reformer but adapted with multiple rounds to improve coverage. Within each bucket, attention is computed exactly, but the total cost is bounded by the bucket sizes. The softpick function, a rectified softmax, then prunes within‑bucket connections, leaving only the strongest ones.

In this paper, we detail the mathematical underpinnings of each component, the integration into a single module, and the implementation within the compression framework. We also discuss the computational complexity, limitations, and potential future improvements.

---

## 2. Method

### 2.1 CUR‑Based Factorised Linear Projections

Given a pretrained weight matrix \( \mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \) and bias \( \mathbf{b} \), the goal is to approximate \( \mathbf{W} \approx \mathbf{C} \mathbf{U} \mathbf{R} \) with \( \mathbf{C} \in \mathbb{R}^{d_{\text{out}} \times r} \), \( \mathbf{U} \in \mathbb{R}^{r \times r} \), and \( \mathbf{R} \in \mathbb{R}^{r \times d_{\text{in}}} \). The rank \( r \) is a hyperparameter (`cur_rank`).

The algorithm uses **leverage scores** to select the most important columns and rows. First, compute the full SVD \( \mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top \). The column leverage scores are the squared row norms of \( \mathbf{V}_{\text{full}} \):

\[
\ell_j^{(c)} = \frac{1}{k} \sum_{i=1}^{k} (\mathbf{V}_{\text{full}})_{j,i}^2 \quad (\text{normalised to sum to }1),
\]

where \( k \) is the number of singular vectors (we use all, but could truncate). Similarly, the row leverage scores are the squared row norms of \( \mathbf{U}_{\text{full}} \). The top‑\( r \) columns and rows with the highest scores are selected:

\[
\mathcal{C} = \operatorname{top}_r \text{ indices by } \ell^{(c)}, \quad
\mathcal{R} = \operatorname{top}_r \text{ indices by } \ell^{(r)}.
\]

Then \( \mathbf{C} = \mathbf{W}_{[:, \mathcal{C}]} \) (size \( d_{\text{out}} \times r \)), \( \mathbf{R} = \mathbf{W}_{[\mathcal{R}, :]} \) (size \( r \times d_{\text{in}} \)). The intersection matrix \( \mathbf{W}_{\mathcal{R}, \mathcal{C}} \in \mathbb{R}^{r \times r} \) is extracted, and its pseudoinverse \( \mathbf{U} = \mathbf{W}_{\mathcal{R}, \mathcal{C}}^+ \) is computed (using `torch.linalg.pinv`).

The factorised layers are then defined as:
- `down`: \( \mathbf{R} \) (maps input to an \( r \)-dimensional space, no bias),
- `up`: \( \mathbf{C} \mathbf{U} \) (maps the \( r \)-dimensional space to output, with bias if the original layer had one).

The forward pass computes \( \mathbf{y} = \text{up}(\text{down}(\mathbf{x})) \). This approximation preserves the original weight’s bias and structure while reducing parameters from \( d_{\text{out}} d_{\text{in}} \) to \( r(d_{\text{out}} + d_{\text{in}}) \).

### 2.2 Locality‑Sensitive Hashing Attention

Standard multi‑head attention yields scores \( \mathbf{S}_{ij} = \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h} \). LSH attention restricts this to pairs that share a hash code under a function \( h: \mathbb{R}^{d_h} \to \mathbb{Z} \). The PHANTOM module uses **angular LSH**: a set of random hyperplane projections followed by a binary encoding.

For each of \( R \) rounds (default 4), a random matrix \( \mathbf{P} \in \mathbb{R}^{d_h \times b} \) is generated with normalised columns. The hash code for a normalised vector \( \hat{\mathbf{q}} = \mathbf{q} / \|\mathbf{q}\| \) is:

\[
h(\mathbf{q}) = \sum_{i=1}^{b} \mathbb{I}((\hat{\mathbf{q}}^\top \mathbf{p}_i) > 0) \cdot 2^{i-1},
\]

where \( \mathbb{I} \) is the indicator. This maps each token to an integer in \( [0, 2^b - 1] \). For a given head, tokens are grouped by this integer (the bucket). Attention is then performed only within each bucket:

\[
\mathbf{S}^{\text{bucket}} = \frac{\mathbf{Q}_{\text{bucket}} \mathbf{K}_{\text{bucket}}^\top}{\sqrt{d_h}}, \quad
\mathbf{A}^{\text{bucket}} = \operatorname{softpick}(\mathbf{S}^{\text{bucket}}),
\]
\[
\mathbf{O}_{\text{query}} = \mathbf{A}^{\text{bucket}} \mathbf{V}_{\text{bucket}}.
\]

The output is scattered back to the original positions. This is repeated for all rounds, and the per‑round outputs are **summed** (not averaged, which may be an oversight). The hashing is performed after L2‑normalisation to make it angular‑based.

The implementation iterates over batch elements and heads, and for each head over queries, using a Python dictionary to map hash codes to key indices. This is highly inefficient but conceptually clear. Padding tokens can be optionally excluded via a `token_valid` mask, which filters the queries and keys before hashing and attention.

### 2.3 Softpick Activation

Within each bucket, the attention weights are computed using the **softpick** function instead of softmax. Softpick is defined as:

\[
\operatorname{softpick}(\mathbf{s})_i = \frac{\operatorname{ReLU}\big(e^{(s_i - m)/\tau} - e^{-m/\tau}\big)}{\sum_j \operatorname{ReLU}\big(e^{(s_j - m)/\tau} - e^{-m/\tau}\big)},
\]

where \( m = \max_j s_j \) and \( \tau \) is a temperature. This rectified version tends to produce sparse outputs—many entries become exactly zero—while maintaining a probability simplex. It is applied to the bucket scores before dropout (if dropout is enabled; the code only applies dropout when softpick is used, which is the default).

---

## 3. Implementation Details

### 3.1 Module Architecture

- **`PHANTOMConfig`**: dataclass holding hyperparameters: `cur_rank` (16), `lsh_rounds` (4), `lsh_bucket_size` (32), `softpick_enabled`, `softpick_temperature`, `dropout`.
- **`FactorisedLinear_CUR`**: Stores `down` (Linear with weight = \( \mathbf{R} \)) and `up` (Linear with weight = \( \mathbf{C}\mathbf{U} \), bias copied if exists). The rank is fixed.
- **`lsh_attention`**: A standalone function that performs the multi‑round LSH attention. It loops over rounds, generates random projections, computes hash codes, and for each batch element and head, iterates over queries to compute bucket‑confined attention using softpick and dropout.
- **`PHANTOMAttention`**: The main module. Constructor takes pretrained attention layer to extract weights for CUR initialisation. Forward pass projects Q, K, V through the factorised layers, computes `token_valid` from mask, calls `lsh_attention`, and projects the output.

### 3.2 Mask Handling

A 2D, 3D, or 4D attention mask is converted to a boolean tensor `token_valid` (True for keep). This mask is passed to `lsh_attention`; when provided, the queries and keys are filtered to valid tokens before hashing. This prevents padding tokens from consuming computation and affecting softmax denominators.

### 3.3 Complexity and Efficiency

The CUR factorisation reduces projection parameters by a factor of roughly \( 2r/(d_{\text{out}}+d_{\text{in}}) \). The LSH attention complexity per round is ideally \( O(N \cdot b_{\text{avg}} d_h) \) where \( b_{\text{avg}} \approx N / 2^b \). For \( b = 32 \), the bucket space is \( 2^{32} \), so buckets are typically tiny; the expected number of comparisons is close to \( O(N) \) in practice. However, the current implementation’s nested Python loops over batch, head, and queries cause a large constant overhead, making it unsuitable for real‑time use without further vectorisation. The softpick adds a small amount of compute but reduces memory for attention weights within buckets.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"phantom"`. Configurable parameters via `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cur_rank` | 16 | Rank for CUR factorisation of Q, K, V, O projections. |
| `lsh_rounds` | 4 | Number of independent hash rounds. |
| `lsh_bucket_size` | 32 | Number of hash bits (bucket size \(2^{32}\)). |
| `softpick_temperature` | 1.0 | Temperature for softpick. |
| `dropout` | 0.1 | Dropout rate (applied within buckets). |

Example injection via framework:
```python
comp_config.attention_type = "phantom"
comp_config.cur_rank = 32
comp_config.lsh_rounds = 6
```

The injection function `inject_phantom` builds the config, constructs the module with pretrained weights, and replaces the layer’s attention.

---

## 5. Discussion

PHANTOM creatively combines two distinct compression paradigms. The CUR decomposition is less common in deep learning than SVD but offers the advantage that the factors \( \mathbf{C} \) and \( \mathbf{R} \) are actual columns/rows of the original weight, which can lead to a more faithful approximation at very low ranks when leverage scores are highly non‑uniform. The LSH attention provides a data‑dependent sparse pattern that adapts to the geometry of the queries and keys. The softpick further sparsifies attention within buckets.

However, the module has several drawbacks in its current form:

- **Sequential loops**: The double loop over batch and head and then over queries (via dictionary) is extremely slow on GPUs. A production version would need to use sort‑and‑chunk techniques to parallelise across hash codes.
- **Summing across rounds**: The LSH function sums outputs from all rounds without averaging, which effectively multiplies the output magnitude by \( R \). This could destabilise training unless the downstream layers can compensate; it should be divided by \( R \).
- **CUR approximation quality**: The deterministic selection of top‑\( r \) leverage scores may not always be optimal; sampling according to the leverage scores (as in the original CUR algorithm) would be more theoretically sound and could improve generalisation.
- **Softpick and masking**: The softpick function is applied per bucket, but the padding mask is handled before hashing (filtering tokens), which is correct. However, the lsh function re‑indexes and may have bugs in the scatter/unsort phase (the code sorts queries but then looks up keys via dictionary without sorting keys, which works because the dictionary maps hash code to key indices; the output is then unsorted using `unsorted_idx = torch.argsort(sorted_q_idx)`. That seems correct, but careful inspection is needed.)
- **Lack of random projection reuse**: The hash matrices are regenerated each round, which is fine. But for inference, one would want deterministic hashing; the random seed should be fixed for reproducibility.

Overall, PHANTOM demonstrates a novel blend of compression and sparse attention and can serve as a research prototype. With optimisation, it could be a viable efficient attention variant.

---

## 6. Conclusion

We have presented PHANTOM, a hybrid attention module that employs CUR‑factorised linear projections and multi‑round angular LSH attention with softpick activation. The CUR decomposition provides parameter efficiency and zero‑shot initialisation, while the LSH mechanism delivers sub‑quadratic sequence‑length scaling. The module is integrated into the DistilBERT compression framework as a drop‑in replacement for standard self‑attention. Future work will focus on vectorising the LSH loops, correcting the round averaging, and exploring probabilistic leverage sampling for improved CUR quality. PHANTOM stands as a creative step toward combining matrix compression with adaptive sparse attention in a single cohesive design.
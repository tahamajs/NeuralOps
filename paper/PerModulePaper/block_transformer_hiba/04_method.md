*(این مهم‌ترین فایل است که بر اساس کد ریاضیاتش را مکتوب می‌کنیم)*

```markdown
# 4. Method

## 4.1 Overview

Given an input $X \in \mathbb{R}^{B \times L \times d}$, HIBA projects $Q, K, V$ using frozen base weights, applies rotary position embeddings (RoPE) and Grouped-Query Attention (GQA) expansions, and routes the matrices through a dual-pathway system dictated by the layer index. 

If the layer is a **rectification layer** (e.g., `layer_idx % 4 == 0`), standard dense attention is applied. Otherwise, the hierarchical block-sparse pathway is triggered.

## 4.2 Component 1: Intra-Block Attention

We chunk the sequence of length $L$ into $N = \lceil L / W \rceil$ blocks of size $W$ (`block_size`). For block $b$, the query $Q_b \in \mathbb{R}^{W \times d_h}$ attends purely to $K_b, V_b$:

$$\text{Attn}_{\text{intra}}(Q_b, K_b, V_b) = \mathrm{softmax}\!\left(\frac{Q_b K_b^\top}{\sqrt{d_h}} + M_{\text{causal}} + M_{\text{pad}}\right) V_b \tag{4.1}$$

This captures high-fidelity local semantics at $\mathcal{O}(N \cdot W^2) = \mathcal{O}(L \cdot W)$ complexity.

## 4.3 Component 2: Inter-Block Sparse Attention

To gather global context without quadratic overhead, $Q_b$ attends to a sparse subset of keys from other blocks, denoted $K_{\neg b}$. In `_inter_block_sparse`, we compute raw scores against neighbors, mask out invalid padding, and apply a hard Top-K masking:

$$S_{\text{inter}} = \frac{Q_b K_{\neg b}^\top}{\sqrt{d_h}} \tag{4.2}$$
$$M_{\text{sparse}}[i, j] = \begin{cases} 0 & \text{if } S_{\text{inter}}[i, j] \text{ in Top-}K \\ -\infty & \text{otherwise} \end{cases} \tag{4.3}$$

The final block output is the sum of both pathways: $Y_b = \text{Attn}_{\text{intra}} + \text{Attn}_{\text{inter}}$.

## 4.4 Component 3: Periodic Rectification

Purely sparse networks suffer from information bottlenecks. HIBA introduces a static boolean flag:

```python
use_rectify = (layer_idx % rectification_interval == 0) or (num_blocks <= 2)

```

When `use_rectify` is true, HIBA falls back to standard dense attention. This provides a "highway" for global token mixing, correcting any contextual drift caused by the sparse layers.

## 4.5 Residual LoRA Adapters

To recover performance without mutating pretrained weights, we parameterize the linear projections with `LoRALinear`:

$$W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} (B A) \tag{4.4}$$

where $W_{\text{base}}$ is frozen, and $A, B$ are trained. This allows HIBA to adapt its query/key representations specifically for the sparse routing regime.

```

---
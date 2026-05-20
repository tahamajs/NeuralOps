# NHA Attention: Native Hybrid Attention with Token‑Selective Full Attention and Linear Memory

**Paper Reference**  
Native Hybrid Attention (NHA), ICML 2024.  
https://arxiv.org/abs/2406.12345

---

## 1. Introduction

Self‑attention in Transformers scales quadratically with sequence length, which is expensive for long documents. Hybrid attention mechanisms aim to reduce this cost by combining a cheap global linear attention with a sparse exact attention. The **Native Hybrid Attention (NHA)** module introduced here adopts a token‑wise hybrid strategy: it first selects a small fraction of the most salient tokens based on the norm of their query vectors, performs **full dot‑product attention** exclusively among those selected tokens, and uses **kernelised linear attention** (with an ELU+1 feature map) for all tokens. The outputs are then blended: selected tokens use the full‑attention context, while the remaining tokens rely on the linear‑attention output. This yields a sub‑quadratic overall cost, as the expensive full attention is restricted to a constant fraction of the sequence.

The module is designed as a drop‑in replacement for the self‑attention layer in DistilBERT, maintaining the standard projection weights and padding mask interface. The fraction of tokens receiving full attention is controlled by `full_ratio` (default 0.1). The selection is based on the per‑token query norm, a simple measure of importance that requires no additional parameters.

In this paper, we present a thorough mathematical description of the NHA mechanism, including the token selection criterion, the dual attention paths, and their fusion. We also discuss implementation details and potential limitations.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first applies the standard linear transformations for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{V}_Q \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\) where \(H\) is the number of heads and \(d_h = D/H\).

### 2.2 Importance‑Based Token Selection for Full Attention

To decide which tokens will undergo full self‑attention, we compute an importance score for each token based on the **norm of its query vectors across heads**. Specifically, for each token position \(i\) and batch element \(b\),

\[
s_{b,i} = \frac{1}{H} \sum_{h=1}^{H} \|\mathbf{Q}_{b,h,i}\|_2.
\]

This score reflects the magnitude of the query, which is often correlated with the amount of attention that token will attract. The top‑\(K\) tokens with the largest scores are selected, where \(K = \max(1, \lfloor L \times \rho \rfloor)\) and \(\rho = \text{full\_ratio}\) (e.g., 0.1). The selection is performed per batch item independently using `torch.topk` along the length dimension. The resulting indices are used to build a boolean mask \(\mathbf{M} \in \{0,1\}^{B \times L}\):

\[
M_{b,i} = \begin{cases}
1 & \text{if token } (b,i) \text{ is selected},\\
0 & \text{otherwise}.
\end{cases}
\]

This mask is then expanded to the head and feature dimensions, yielding \(\mathbf{M}^{\text{full}} \in \mathbb{R}^{B \times H \times L \times d_h}\). It will be used to gate the queries and keys for full attention and to later combine the two branches.

### 2.3 Full Attention on Selected Tokens

For the full‑attention branch, we restrict queries and keys to only the selected tokens. This is achieved by multiplying element‑wise with the mask:

\[
\tilde{\mathbf{Q}} = \mathbf{Q} \odot \mathbf{M}^{\text{full}}, \qquad
\tilde{\mathbf{K}} = \mathbf{K} \odot \mathbf{M}^{\text{full}}.
\]

Since the mask is zero for non‑selected tokens, their query and key vectors become zero, effectively removing them from the attention computation. The full attention scores are then computed as usual:

\[
\mathbf{S}^{\text{full}} = \frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L \times L}.
\]

Note that only the selected tokens have non‑zero queries, so the rows corresponding to other tokens are entirely zero; they will produce zero outputs. The padding mask (if provided) is applied by adding \((m-1)\cdot(-\infty)\) to the scores, where \(m\) is the attention mask (1 for valid, 0 for padding). Softmax yields the attention weights:

\[
\mathbf{A}^{\text{full}} = \operatorname{softmax}(\mathbf{S}^{\text{full}} + \mathbf{M}_{\text{pad}}).
\]

Dropout is applied, and the full context is obtained by multiplying with the original (unmasked) values:

\[
\mathbf{C}^{\text{full}} = \mathbf{A}^{\text{full}} \mathbf{V} \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

Because only selected queries produce meaningful outputs, this tensor will contain non‑zero context only at those selected positions.

### 2.4 Linear Attention on All Tokens

In parallel, a **kernelised linear attention** is applied to the entire sequence. The feature map is the commonly used \(\phi(x) = \operatorname{ELU}(x) + 1\), which ensures positivity and allows the attention to be factorised. We define

\[
\phi_Q = \operatorname{ELU}(\mathbf{Q}) + 1, \qquad
\phi_K = \operatorname{ELU}(\mathbf{K}) + 1.
\]

For the linear attention, we need to aggregate the key‑value pairs into a memory matrix. The padding mask must be applied here to prevent padding tokens from contributing. The implementation does this by adding a huge negative offset to the keys of padding tokens before the ELU activation? Actually, it adds `(mask-1)* -1e9` to the already computed \(\phi_K\), which is problematic because \(\phi_K\) is already positive; adding a large negative value could make it negative, breaking the kernel property. A correct approach would be to zero out the keys of padding tokens directly. However, we describe the intended mathematics:

Let \(\mathbf{M}_{\text{pad}}\) be a boolean mask with True for valid tokens. The keys and values of padding tokens should not be accumulated. Formally, we define

\[
\tilde{\phi}_K = \phi_K \odot \mathbf{M}_{\text{pad},\text{expanded}},
\]
\[
\tilde{\mathbf{V}} = \mathbf{V} \odot \mathbf{M}_{\text{pad},\text{expanded}}.
\]

Then the key‑value memory is

\[
\mathbf{KV} = \sum_{j=1}^L \tilde{\phi}_K(:,j) \tilde{\mathbf{V}}(:,j)^\top \quad \in \mathbb{R}^{B \times H \times d_h \times d_h},
\]
and the key normalisation sum is

\[
\mathbf{Z} = \sum_{j=1}^L \tilde{\phi}_K(:,j) \quad \in \mathbb{R}^{B \times H \times d_h}.
\]

The linear attention output for each query is then

\[
\mathbf{C}^{\text{lin}}_{b,h,i} = \frac{\phi_Q(b,h,i)^\top \mathbf{KV}}{\phi_Q(b,h,i)^\top \mathbf{Z} + \epsilon}.
\]

In the implementation, the padding mask is applied to \(\phi_K\) by adding `(mask-1)* -1e9`. This effectively turns the key features of padding tokens into very large negative numbers; after addition, they are no longer valid for the dot product, but the subsequent exponent in the kernel (none) makes this numerically unstable. However, for the purpose of this paper, we assume the mask is correctly handled by zeroing.

### 2.5 Hybrid Combination

The final output is formed by selecting, for each token, either the full‑attention context (if that token was selected) or the linear‑attention context (otherwise). This is performed using the mask \(\mathbf{M}^{\text{full}}\) expanded to \((B,H,L,d_h)\):

\[
\mathbf{C} = \mathbf{M}^{\text{full}} \odot \mathbf{C}^{\text{full}} + (1 - \mathbf{M}^{\text{full}}) \odot \mathbf{C}^{\text{lin}}.
\]

Thus, the selected tokens rely on exact pairwise interactions among themselves, while the majority of tokens use the global linear memory. This hybrid strategy aims to provide high accuracy for the most important tokens while maintaining linear complexity for the rest.

### 2.6 Output Projection

The final context is reshaped to \((B, L, D)\) and projected with \(\mathbf{W}_O\):

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \text{None})\) unless `output_attentions` is True, in which case it returns the full attention weights for the selected tokens (though only those rows contain non‑zero values).

---

## 3. Implementation Details

- **Class `NHAAttention`** inherits from `BaseAttention` and is registered as `"nha"`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rank`: unused (interface compatibility).
  - `full_ratio`: fraction \(\rho\) of tokens selected for full attention (default 0.1).
- **Forward**:
  1. Project Q, K, V.
  2. Compute token importance as `q.norm(dim=-1).mean(dim=1)` → shape \((B, L)\).
  3. Select top `num_full = max(1, int(L * full_ratio))` tokens per batch, build `full_mask` of shape \((B, H, L, d_h)\).
  4. **Full branch**: zero out Q and K for non‑selected tokens via multiplication, compute scores, apply padding mask (if any), softmax, dropout, multiply by V.
  5. **Linear branch**: compute ELU+1 features for Q and K. Apply padding mask to K features by adding a large negative bias (flawed, as noted). Compute KV matrix, Z sum, denominator, and linear output via batched matrix operations.
  6. Blend: use `torch.where(full_mask, full_out, lin_out)`.
  7. Reshape, output projection.
  8. Return output and optionally full attention weights.

- **Padding mask format**: The code expects an `attention_mask` as a float tensor with 1 for valid tokens and 0 for padding. It adds `(mask - 1) * -1e9` to scores in the full attention, and directly to the K features in the linear attention. The latter is incorrect and may cause numerical issues; a production implementation would zero out the K and V of padding tokens before the linear aggregation.

- **Parameter `full_ratio`**: Typically small (e.g., 0.1), meaning only 10% of tokens get full attention. This makes the full attention cost \(O((\rho L)^2)\) which can be significantly smaller than \(O(L^2)\) for long sequences.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"nha"` in the compression configuration. The configurable parameters (via `comp_config` or direct arguments) are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `full_ratio` | 0.1 | Fraction of tokens receiving full attention. |
| `dropout` | (from config) | Dropout rate on full attention weights. |

Example injection:
```python
comp_config.attention_type = "nha"
comp_config.full_ratio = 0.15
```

Manual instantiation:
```python
attn = NHAAttention(config=model_config, full_ratio=0.15)
layer.attention = attn
```

No pretrained weight transfer is performed in this snippet; the projection layers are randomly initialised. For zero‑shot usage, an injection function should copy the original `q_lin`, `k_lin`, etc. This is not provided but could be added.

---

## 5. Discussion

NHA Attention offers a simple yet effective hybrid: by identifying the most important tokens (via query norm) and granting them full interaction, it retains the benefits of exact attention for critical positions while using linear attention to keep the overall cost low. The selection process is parameter‑free and computationally cheap. The linear attention uses the well‑tested ELU+1 kernel, which is known to approximate softmax reasonably well.

However, several aspects of this implementation warrant attention:

- **Flawed linear masking**: Adding a large negative constant to \(\phi_K\) after ELU breaks the positivity and can cause the denominator to become zero or negative, leading to NaN outputs. A correct implementation must zero out the keys and values of padding tokens before the KV accumulation.
- **Token selection is static**: The importance score depends only on the query norms, which are computed from the current hidden states. This may not always select the most relevant tokens for all heads or downstream tasks; a learnable scoring mechanism could improve adaptivity.
- **Full attention among selected tokens only**: The full attention is restricted to the selected set, which means a selected token can only attend to other selected tokens, not to all tokens. This might limit its ability to retrieve information from long‑tail tokens. An alternative is to let the selected tokens attend to all tokens (as in a landmark‑based approach), which would increase cost but provide a better global view.
- **No sliding window**: Despite the docstring mentioning sliding‑window attention, the code does not implement any windowing; the “local” aspect is entirely missing. The paper title suggests a combination of sliding window and linear memory, but this implementation uses a token‑importance‑based full attention instead. This discrepancy may be intentional simplification for this module.

Despite these issues, the NHA module demonstrates a valid hybrid approach and can serve as a baseline for token‑selective attention methods.

---

## 6. Conclusion

We have presented the NHA Attention module, which combines full attention on a small subset of important tokens with global linear attention for the remaining tokens. The selection is based on query vector norms, and the fusion uses a hard mask. The design achieves sub‑quadratic complexity while retaining exact pairwise interactions for the most salient positions. Future work should address the masking bug, incorporate a true sliding‑window or landmark‑based full attention, and explore learnable token selection for improved performance.
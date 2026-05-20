# ViG Attention: Vision GNN‑Inspired Grid Linear Attention with Gated Fusion

**Abstract**

We present **ViG Attention**, a novel efficient self‑attention module inspired by Vision GNN architectures. The module partitions the input sequence into a rectangular grid of dimensions \(H \times W\) and performs two independent linear attentions: one along each row (width‑wise) and one along each column (height‑wise). Both linear attentions use a simple ELU+1 kernel to achieve linear complexity in the sequence length, avoiding the quadratic cost of full attention. The outputs of the row and column attention branches are then adaptively fused via a learned head‑wise gate that produces two scalars per token, controlling the contribution of each branch. The final representation is obtained after de‑padding the grid back to the original sequence length. This design captures both horizontal and vertical contextual interactions while keeping the computational cost at \(O(N \sqrt{N} d)\) if the grid is approximately square. The module is designed as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, integrating seamlessly into the compression framework via the registry. This paper provides a rigorous mathematical description of the grid formation, the dual linear attention, the gating mechanism, and the overall architecture.

---

## 1. Introduction

Standard multi‑head self‑attention scales quadratically with the sequence length \(N\), which is prohibitive for long documents. Several efficient attention mechanisms have been proposed that reduce this cost, often by restricting the attention pattern to a local window or by using kernelised linear attention. An alternative approach, popular in Vision Transformers and Graph Neural Networks, is to impose a structured grid on the input and perform attention along one axis at a time. The **ViG Attention** module adapts this idea to language sequences: it reshapes the token sequence into a 2D grid (with dimensions \(H \times W \approx \sqrt{N} \times \sqrt{N}\)), and then applies **linear attention separately along rows and columns**. The row attention captures dependencies across the width of the grid, while the column attention captures dependencies across the height. By decomposing the attention into these two orthogonal axes, the module effectively approximates full 2D attention while maintaining a computational complexity of \(O(N H d + N W d)\) instead of \(O(N^2 d)\), which for a square grid becomes \(O(N^{3/2} d)\).

The attention itself is further linearised using the ELU+1 kernel (Katharopoulos et al., 2020), which avoids materialising an attention matrix even within each axis. Thus, the overall complexity is sub‑quadratic. A gating mechanism learns to combine the row and column outputs, allowing the model to dynamically weight their contributions per token and per head.

The module is named “ViG” likely as a nod to Vision GNNs, which also use graph‑based and grid‑based message passing. However, it is applied here as a general‑purpose efficient attention replacement for DistilBERT, with no visual components. The module is registered in the compression framework under the key `"vig"` and can be injected into any DistilBERT layer.

In this paper, we detail the grid formation, the dual linear attention, the gated fusion, and the de‑padding procedure. We also discuss the complexity and potential applications of the method.

---

## 2. Method

### 2.1 Grid Formation

Given an input sequence of hidden states \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), the module first determines a suitable grid size to reshape the sequence into a 2D layout. The height \(H\) is chosen as the integer square root of \(N\):

\[
H = \lfloor \sqrt{N} \rfloor.
\]

The width \(W\) is then computed as the smallest integer such that \(H \times W \ge N\):

\[
W = \lceil N / H \rceil.
\]

The total number of cells in the grid is \(P = H \times W\). If \(P > N\), the input is padded with zeros along the sequence dimension to length \(P\). The padded sequence is then reshaped to \((B, H, W, D)\), i.e., a 2D spatial layout. The padding tokens are effectively masked by the subsequent operations (their features are zero, so they contribute nothing to the attention after the ELU activation, but they still participate in the normalisation; we will discuss this later).

### 2.2 Multi‑Head Projections

The padded input is projected to queries, keys, and values for all heads:

\[
\mathbf{Q} = \operatorname{Proj}_Q(\mathbf{X}_{\text{pad}}), \quad
\mathbf{K} = \operatorname{Proj}_K(\mathbf{X}_{\text{pad}}), \quad
\mathbf{V} = \operatorname{Proj}_V(\mathbf{X}_{\text{pad}}),
\]

each of shape \((B, P, H d_h)\) where \(H\) is the number of heads and \(d_h = D/H\). They are then reshaped to multi‑head format \((B, H_{\text{heads}}, P, d_h)\) and further rearranged into the grid layout \((B, H_{\text{heads}}, H, W, d_h)\). For clarity, we use \(H_h\) to denote the number of heads.

### 2.3 Learned Head‑Wise Gates

From the padded input (before projection), a gate network computes two scalar coefficients per token and per head. A linear layer \(\mathbf{W}_g \in \mathbb{R}^{D \times 2 H_h}\) (with bias) projects the padded hidden states to a \(2 H_h\)-dimensional vector, which is split into two sets of \(H_h\) values. A sigmoid activation yields:

\[
\boldsymbol{\gamma}_r, \boldsymbol{\gamma}_c = \operatorname{split}(\sigma(\mathbf{X}_{\text{pad}} \mathbf{W}_g)) \in \mathbb{R}^{B \times P \times H_h \times 1} \text{ each}.
\]

These gates will be used to weight the row and column outputs, respectively, after reshaping them to the grid.

### 2.4 Row Attention (Along Width)

For the row attention, we treat each row as an independent sequence of length \(W\). The tensors are reshaped and transposed as follows:

- Queries, keys, values are permuted from \((B, H_h, H, W, d_h)\) to \((B, H_h, W, H, d_h)\).
- They are then merged into shape \((B, H_h W, H, d_h)\) to process all rows in parallel. Actually, the code does:
  ```
  q = q.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)   # (B, Hd, W, H, Dh)
  q_row = q.reshape(B, Hd * W, H, Dh).transpose(2, 3)     # (B, Hd*W, Dh, H)
  ```
  Let's step through: after permute, shape is (B, Hd, W, H, Dh). Reshape to (B, Hd*W, H, Dh) merges heads and width into the batch dimension. Then transpose(2,3) gives (B, Hd*W, Dh, H). So each row of height H becomes a sequence of length H with feature dimension Dh. Wait, earlier they said row attention is along rows. The grid is (H, W) where H is height, W is width. Row attention should be along the width dimension. But the code reshapes to treat H as the sequence length. Let's examine carefully:

  Original grid shape: (B, Hd, H, W, Dh). They permute to (B, Hd, W, H, Dh). So the dimensions are (batch, heads, width, height, feature). Then they reshape to (B, Hd*W, H, Dh). So they merge batch*heads*width into a single batch dimension, leaving a sequence of length H (height) and feature Dh. Then they transpose to (B, Hd*W, Dh, H) and later do linear attention with einsum 'b m n d, b m n e -> b m d e', where m is the merged batch, n is sequence length, d/e are features. So they are treating the height dimension as the sequence and performing linear attention over it. That means "row attention" is actually attending along the height of the grid (i.e., across different rows). Conversely, "column attention" attends along the width. The naming is swapped relative to the intuitive notion; row attention in the code means attention over the row index (height), column attention means attention over the column index (width). We'll describe it as "attention over the H axis" (height) and "attention over the W axis" (width) to avoid confusion.

  Let's re-derive: The two axes are H (number of rows) and W (number of columns). One attention will aggregate information along the H dimension (i.e., each column position attends to all other positions in the same column), and the other along the W dimension (each row position attends to all other positions in the same row). That matches "column attention" and "row attention" in the conventional sense. However, the code's variable names `out_row` and `out_col` may be mislabeled. We'll adopt the following: **Height‑axis attention** (over rows) and **Width‑axis attention** (over columns). The code does:

  - Height‑axis: `q_row`, `k_row`, `v_row` derived after `q.reshape(B, Hd * W, H, Dh)`, so they process sequences of length H, each of width W? Actually the sequence length is H, and the feature dimension is Dh. So this is attending along the height (each column of the grid). So it's column‑wise attention. Let's call it **vertical attention**.

  - Width‑axis: `q_col_flat` etc. from `q_col_flat = q_col.reshape(B * Hd * H, W, Dh)`, so sequence length is W. That's horizontal attention.

  So we'll define: vertical attention (over H) and horizontal attention (over W). The gates `gate_row` and `gate_col` correspond to vertical and horizontal respectively. But the code assigns `gate_row` to `out_row` (vertical) and `gate_col` to `out_col` (horizontal). We'll just follow the code's naming: row‑attention is vertical (over H), column‑attention is horizontal (over W). The paper description should align with the implementation.

  I'll call them **axis‑H attention** and **axis‑W attention**.

### 2.5 Linear Attention with ELU+1 Kernel

Both axis attentions use the same kernelised linear attention formula. For a general sequence of length \(S\) with queries \(\mathbf{Q}\), keys \(\mathbf{K}\), values \(\mathbf{V}\) (all of shape \((B', S, d)\)), the feature map is:

\[
\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1.
\]

The key–value memory and normalisation are:

\[
\mathbf{KV} = \sum_{j=1}^S \phi(\mathbf{k}_j) \mathbf{v}_j^\top \in \mathbb{R}^{B' \times d \times d}, \qquad
\mathbf{Z} = \sum_{j=1}^S \phi(\mathbf{k}_j) \in \mathbb{R}^{B' \times d}.
\]

The output for each query is:

\[
\mathbf{o}_i = \frac{\phi(\mathbf{q}_i)^\top \mathbf{KV}}{\phi(\mathbf{q}_i)^\top \mathbf{Z} + \epsilon}.
\]

In batched form, this is computed using `einsum`. For axis‑H, the merged batch is \(B' = B \times H_h \times W\), and sequence length \(S = H\). For axis‑W, the merged batch is \(B' = B \times H_h \times H\), and sequence length \(S = W\).

### 2.6 Axis‑H (Vertical) Attention

- Queries, keys, values reshaped: \(\mathbf{Q}_{v}, \mathbf{K}_{v}, \mathbf{V}_{v} \in \mathbb{R}^{B \times (H_h W) \times d_h \times H}\) after transpose? The code gets to shape `(B, Hd*W, Dh, H)` for queries, then applies ELU+1 and does linear attention with `einsum('b m n d, b m n e -> b m d e', k_row_lin, v_row)` where `m` is the merged batch dimension (B*Hd*W), `n` is sequence length H, `d` and `e` are features Dh. Then output is `(B, Hd*W, Dh, H)` which is reshaped back to `(B, Hd, W, H, Dh)` and then permuted to `(B, Hd, H, W, Dh)`.

### 2.7 Axis‑W (Horizontal) Attention

- Queries, keys, values are first permuted to `(B, Hd, H, W, Dh)`? Actually from `(B, Hd, W, H, Dh)` (after the first permute for vertical attention), they need to be rearranged for horizontal. The code starts from `q = q.reshape(B, Hd, H, W, Dh).permute(0,1,3,2,4)` which gave `(B, Hd, W, H, Dh)`. To get horizontal, they do `q_col = q.permute(0,1,2,4,3)` → `(B, Hd, W, Dh, H)`, then reshape to `(B*Hd*H, W, Dh)`. Then linear attention with sequence length W. Output reshaped back to `(B, Hd, H, W, Dh)`.

### 2.8 Gated Fusion

The two outputs are combined using the learned gates. The vertical gate (called `gate_row` in code) is reshaped to `(B, Hd, H, W, 1)`, and the horizontal gate (`gate_col`) similarly. The final output is:

\[
\mathbf{O}_{\text{grid}} = \boldsymbol{\gamma}_v \odot \mathbf{O}_{v} + \boldsymbol{\gamma}_h \odot \mathbf{O}_{h},
\]

where \(\mathbf{O}_{v}\) is the vertical attention output (shape \((B, H_h, H, W, d_h)\)) and \(\mathbf{O}_{h}\) is the horizontal attention output (same shape). The gates sum to? They are independent sigmoid outputs, not constrained to sum to 1; they can both be close to 1, so the fusion is a weighted average with two independent weights.

### 2.9 De‑padding and Output Projection

The fused grid output is reshaped back to \((B, P, D)\). If padding was applied, the first \(N\) positions are retained, and the rest are discarded. Finally, a linear output projection \(\mathbf{W}_O\) is applied to obtain the result \(\mathbf{Y} \in \mathbb{R}^{B \times N \times D}\).

The module returns \((\mathbf{Y}, \text{None})\) (attention weights are not returned).

### 2.10 Complexity

The vertical attention cost: \(O(B \cdot H_h \cdot W \cdot H \cdot d_h^2) = O(B \cdot D \cdot H \cdot d_h)\) because \(H_h \cdot d_h = D\) and \(W \approx N/H\). So vertical cost \(\approx O(B D N d_h / H)\). Horizontal cost: \(O(B \cdot D \cdot W \cdot d_h) \approx O(B D N d_h / W)\). With \(H \approx W \approx \sqrt{N}\), each axis is \(O(B D \sqrt{N} d_h)\), and total is \(O(B D \sqrt{N} d_h)\), which is sub‑quadratic. However, because the feature dimension in the linear attention is \(d_h\) (not reduced), the constant factor is small.

---

## 3. Implementation Details

- **Class `ViGAttention`** inherits from `BaseAttention` and is registered as `"vig"`.
- **Constructor**:
  - `config`: provides `hidden_size`, `num_attention_heads`, `dropout` (not used in this code—dropout is not applied).
  - `rank`: ignored.
  - Creates `q_proj`, `k_proj`, `v_proj`, `out_proj`, and `gate_net` (output size \(2 \times \text{num\_heads}\)).
- **Forward**:
  1. Determine grid size \(H = \lfloor\sqrt{N}\rfloor\), \(W = \lceil N/H\rceil\).
  2. Pad input to length \(P = H \times W\) with zeros.
  3. Project Q, K, V, and compute gates from padded input.
  4. Reshape to grid and perform axis‑H and axis‑W linear attentions as described.
  5. Fuse with gates.
  6. Reshape back, remove padding, apply output projection.
  7. Return `(output, None)`.
- **Mask handling**: The module does **not** accept or use an attention mask. Padding tokens within the padded grid are not explicitly masked; they are processed as zeros. The ELU+1 kernel will output 1 for zero input (since ELU(0)=0, +1=1), which means the padding positions will contribute a constant vector to the key‑value memory, potentially biasing the normalisation. This could be a limitation; a proper implementation would set padding features to a large negative value before ELU to ignore them, or explicitly mask them. The current version assumes no padding in the original input (or that the zero padding is harmless).

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"vig"` in the compression configuration. No additional hyperparameters are required. Example:

```python
comp_config.attention_type = "vig"
```

The module does not initialise from pretrained weights; the linear layers are randomly initialised. An injection function (not shown) could copy the original Q, K, V, O weights if zero‑shot performance is desired. Without it, the module must be fine‑tuned.

---

## 5. Discussion

ViG Attention offers an interesting alternative to standard full attention by leveraging a grid decomposition. The two linear attention axes capture dependencies along both spatial dimensions, which can be seen as a form of axial attention (as in Axial Transformers). However, the use of a rectangular grid may not align with the natural sequential structure of language; the arbitrary padding to a rectangle might disrupt linguistic coherence. Nevertheless, for tasks where the sequence can be interpreted as a 2D layout (e.g., document layouts, images), this method is particularly apt. For pure text, the grid acts as a fixed local‑ish connectivity pattern that approximates full attention at a reduced cost.

The ELU+1 linear attention further reduces complexity, but as with all linear attention methods, it may not capture sharp attention patterns as well as softmax. The gating mechanism gives the model some flexibility to mix the two axes, but the gates are computed from the raw input, not from the attention outputs themselves, which may limit their adaptivity.

The main limitations are:
- **No padding mask**: The module cannot handle variable‑length sequences with padding tokens unless the caller pads explicitly and manages masking. The current simple zero‑padding may introduce artifacts.
- **Fixed grid shape**: The grid size is determined solely by sequence length, not by content. This may be suboptimal for sequences where a different aspect ratio would capture better dependencies.
- **No dropout**: Dropout is not applied to attention weights, which could lead to overfitting.
- **Gate design**: The gates are per‑token and per‑head but not constrained, which might cause training instability. A softmax across the two gate values could be more stable.

Despite these issues, ViG Attention provides a novel and computationally efficient attention scheme that can be easily integrated into the compression framework. It demonstrates the potential of combining structural priors (grid) with linearised attention for long sequence processing.

---

## 6. Conclusion

We have presented ViG Attention, a grid‑based linear attention module that decomposes the sequence into rows and columns, performs kernelised linear attention along each axis, and fuses the results with learned gates. The approach achieves sub‑quadratic complexity and is designed as a drop‑in replacement for DistilBERT’s self‑attention. The module enriches the library of efficient attention mechanisms available in the compression framework and offers a basis for future exploration of structured, axis‑wise attention patterns in Transformers. Future work will address the masking limitations and explore dynamic grid allocation based on content.**ViG Attention: Vision GNN‑Inspired Grid Linear Attention with Gated Fusion**

**Abstract**
We present **ViG Attention**, a novel efficient self‑attention module inspired by Vision GNN architectures. The module partitions the input sequence into a rectangular grid of dimensions \(H \times W\) and performs two independent linear attentions: one along the height axis (vertical, attending across rows) and one along the width axis (horizontal, attending across columns). Both linear attentions use a simple ELU+1 kernel to achieve linear complexity in the sequence length, avoiding the quadratic cost of full attention. The outputs of the two branches are then adaptively fused via a learned head‑wise gate that produces two scalars per token, controlling the contribution of each axis. The final representation is obtained after de‑padding the grid back to the original sequence length. This design captures both vertical and horizontal contextual interactions while keeping the computational cost at \(O(N \sqrt{N} d)\) when the grid is approximately square. The module is designed as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, integrating seamlessly into the compression framework via the registry. This paper provides a rigorous mathematical description of the grid formation, the dual linear attention, the gating mechanism, and the overall architecture.

**1. Introduction**

Standard multi‑head self‑attention scales quadratically with the sequence length \(N\), making it costly for long documents. Many efficient attention methods have been proposed, often by restricting the attention pattern or using kernelised linear attention. An alternative, inspired by Vision Transformers and Graph Neural Networks, is to impose a structured 2D grid on the input and perform attention along each axis independently. The **ViG Attention** module adapts this idea to language sequences: it reshapes the token sequence into a grid (height \(H\), width \(W\), with \(H \times W \ge N\)), and then applies **kernelised linear attention separately along the height and width dimensions**. This axis‑wise decomposition approximates full 2D attention while reducing the complexity from \(O(N^2 d)\) to \(O(N H d + N W d)\), which for a square grid is roughly \(O(N^{3/2} d)\). The linear attention itself uses the ELU+1 feature map, avoiding the materialisation of an attention matrix even within each axis. A gating mechanism learns to combine the two axial outputs per token and per head.

The module is named “ViG” (likely a nod to Vision GNNs) but is applied as a general‑purpose efficient attention replacement for DistilBERT. It is registered in the compression framework under the key `"vig"`. In this paper, we detail the grid formation, the dual linear attention, the gated fusion, and the de‑padding procedure, together with a discussion of its computational properties and limitations.

**2. Method**

**2.1 Grid Formation**

Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), the module first determines a suitable grid size. The height \(H\) is chosen as the integer square root of \(N\):
\[ H = \lfloor \sqrt{N} \rfloor. \]
The width \(W\) is then the smallest integer such that \(H \times W \ge N\):
\[ W = \lceil N / H \rceil. \]
The total number of grid cells is \(P = H \times W\). If \(P > N\), the input is padded with zeros along the sequence dimension to length \(P\); the padded sequence is then reshaped to \((B, H, W, D)\).

**2.2 Multi‑Head Projections and Gates**

The padded input is linearly projected to queries, keys, and values for all heads (the number of heads is \(H_h\), head dimension \(d_h = D / H_h\)):
\[ \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times P \times D} \quad \text{(then reshaped to } (B, H_h, P, d_h) \text{ and further to } (B, H_h, H, W, d_h)). \]
A gate network computes two scalar coefficients per token and per head from the padded input:
\[ \mathbf{g} = \sigma(\mathbf{X}_{\text{pad}} \mathbf{W}_g) \in \mathbb{R}^{B \times P \times 2 H_h}, \]
which is split and reshaped into \(\boldsymbol{\gamma}_v, \boldsymbol{\gamma}_h \in \mathbb{R}^{B \times H_h \times H \times W \times 1}\) for the vertical (axis‑H) and horizontal (axis‑W) branches respectively.

**2.3 Axis‑H (Vertical) Linear Attention**

For vertical attention, we treat each column of the grid as an independent sequence. The tensors are reshaped to merge the batch, heads, and width dimensions, yielding sequences of length \(H\) with feature dimension \(d_h\):
\[ \mathbf{Q}_v, \mathbf{K}_v, \mathbf{V}_v \in \mathbb{R}^{B \times (H_h W) \times d_h \times H} \quad (\text{after appropriate transpositions}). \]
The ELU+1 feature map \(\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1\) is applied to queries and keys. Linear attention is then performed:
\[ \mathbf{KV}_v = \sum_{j=1}^{H} \phi(\mathbf{K}_{v,j}) \mathbf{V}_{v,j}^\top \in \mathbb{R}^{B' \times d_h \times d_h}, \qquad
\mathbf{Z}_v = \sum_{j=1}^{H} \phi(\mathbf{K}_{v,j}) \in \mathbb{R}^{B' \times d_h}, \]
\[ \mathbf{O}_v = \frac{\phi(\mathbf{Q}_v) \mathbf{KV}_v}{\phi(\mathbf{Q}_v) \mathbf{Z}_v + \epsilon} \in \mathbb{R}^{B' \times d_h \times H}, \]
where \(B' = B \times H_h \times W\). The output is then reshaped back to \((B, H_h, H, W, d_h)\).

**2.4 Axis‑W (Horizontal) Linear Attention**

For horizontal attention, each row becomes a sequence of length \(W\). The tensors are reshaped to merge batch, heads, and height:
\[ \mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h \in \mathbb{R}^{(B \cdot H_h \cdot H) \times W \times d_h}. \]
The same ELU+1 linear attention is applied, producing an output reshaped to \((B, H_h, H, W, d_h)\).

**2.5 Gated Fusion and Output**

The two axial outputs are combined using the learned gates:
\[ \mathbf{O}_{\text{grid}} = \boldsymbol{\gamma}_v \odot \mathbf{O}_v + \boldsymbol{\gamma}_h \odot \mathbf{O}_h \in \mathbb{R}^{B \times H_h \times H \times W \times d_h}. \]
The grid is then flattened to \((B, P, D)\). If padding was applied, the first \(N\) positions are kept and the padded part discarded. Finally, a linear output projection \(\mathbf{W}_O\) yields the result \(\mathbf{Y} \in \mathbb{R}^{B \times N \times D}\).

**2.6 Complexity**

The vertical attention costs \(O(B \cdot D \cdot H \cdot d_h)\) (since \(H_h W d_h \approx D \cdot (N/H)\)) and the horizontal attention costs \(O(B \cdot D \cdot W \cdot d_h)\). With \(H \approx W \approx \sqrt{N}\), the total cost is \(O(B D \sqrt{N} d_h)\), which is sub‑quadratic and significantly lower than full attention for long sequences.

**3. Implementation Details**

- **Class `ViGAttention`** inherits from `BaseAttention` and is registered as `"vig"`.
- **Grid size**: `H = int(N**0.5)`, `W = (N + H - 1) // H`.
- **Padding**: zero padding is used; no explicit masking of padding tokens is performed inside the attention, which may cause a slight bias in the key‑value memory.
- **Gates**: The gate network is a linear layer outputting \(2 \times \text{num\_heads}\) values; sigmoid activation yields independent weights.
- **Linear attention**: All `einsum` operations are batched over the merged dimensions; the denominator uses an epsilon of \(10^{-8}\).
- **Dropout**: Not applied in the current implementation.
- **Injection**: The module is injected via the registry key `"vig"`; no pretrained weight transfer is performed in this snippet.

**4. Usage and Configuration**

Set the attention type to `"vig"` in the compression configuration:
```python
comp_config.attention_type = "vig"
```
No additional hyperparameters are required. The module will be randomly initialised; for zero‑shot use, an external injection function should copy the original Q, K, V, O weights.

**5. Discussion**

ViG Attention offers an efficient, axis‑wise decomposition of attention that is particularly appealing for very long sequences. The grid structure introduces a locality bias: tokens that are far apart in the sequence can only interact indirectly through multiple layers (since they are separated in both grid dimensions). The dual linear attention preserves a degree of global mixing because all tokens in the same row/column interact, but tokens in different rows and columns only share information via the gates and subsequent layers. The method is reminiscent of Axial Transformers, but replaces the softmax attention on each axis with a cheaper linearised variant.

Key limitations include:
- **No padding mask**: Padded zeros contribute a constant vector (ELU(0)+1 = 1) to the memory, which can distort the normalisation. A proper implementation should set pad positions to a large negative value before ELU, or explicitly mask them.
- **Fixed square grid**: The aspect ratio is fixed to roughly square, which may not align with the natural sequential structure of text. For very long sequences, alternative aspect ratios could be more suitable.
- **No dropout**: The lack of dropout on attention weights may lead to overfitting.
- **Gate design**: The gates are computed from the raw input and not constrained; they could be replaced by a softmax over the two axes or conditioned on the attention outputs for better adaptivity.

Despite these issues, the module provides a clean and computationally efficient alternative to standard attention and enriches the collection of methods in the compression framework. Its grid‑based approach may inspire further research into structured, multi‑axis attention for language tasks.

**6. Conclusion**

We have presented ViG Attention, a grid‑based linear attention module that decomposes the sequence into rows and columns, performs kernelised attention along each axis, and fuses the results with learned gates. The design achieves sub‑quadratic complexity and is integrated as a drop‑in replacement for DistilBERT’s self‑attention. Future work will address the masking issues and explore dynamic grid sizing to better adapt to input length and content. ViG Attention demonstrates the viability of axis‑wise decomposition for efficient sequence modelling.
# TPAttention: Tensor Product Attention with Factorized Query‑Key Projections

**Abstract**

We present **TPAttention** (Tensor Product Attention), a novel self‑attention module that replaces the standard linear query and key projections with a **tensor product of two low‑rank factorised features**. Instead of mapping the input directly to a query vector of dimension \(d_h\), the module projects the hidden state to two smaller vectors per head, each of rank \(r\), and then computes their outer product to form a query (and key) of dimension \(r^2\). The result is a structured, higher‑dimensional representation for queries and keys that captures multiplicative interactions between the factorised features. The values are projected in the usual way, and the attention scores are computed as scaled dot‑products over the \(r^2\)‑dimensional query and key vectors. This design increases the expressive capacity of the attention head without a proportional increase in the number of parameters, as the two projections are low‑rank. The module is designed as a drop‑in replacement for the self‑attention layer in DistilBERT and is registered within the compression framework. This paper provides a rigorous mathematical formulation of the tensor product attention, its parameterisation, complexity, and potential benefits.

---

## 1. Introduction

Standard multi‑head self‑attention computes queries and keys as linear transformations of the input, resulting in vectors of dimension \(d_h\) (the head dimension). The dot product \(\mathbf{q}^\top \mathbf{k}\) captures linear interactions between the query and key features. To enrich the representation, one may introduce multiplicative interactions, where the query and key are formed by tensor products of factorised components. This is the central idea behind **Tensor Product Attention (TPA)**. By splitting each query and key into two low‑rank factors and taking their outer product, TPA constructs a query vector that encodes pairwise interactions between the factorised features. This yields a higher effective dimension (\(r^2\) instead of \(d_h\)) while keeping the number of parameters proportional to \(r \cdot d_{\text{model}}\) rather than \(r^2 \cdot d_{\text{model}}\), because the factors are shared across heads and the outer product is computed on the fly.

The TPA module introduced here implements this concept. For each head, the hidden state is projected to two factor vectors \(\mathbf{q}_1, \mathbf{q}_2 \in \mathbb{R}^{B \times L \times H \times r}\), and the full query is obtained as \(\mathbf{Q} = \mathbf{q}_1 \otimes \mathbf{q}_2\), giving a vector of size \(r^2\). The same is done for keys. Values retain the standard projection of dimension \(d_h\). Attention is then performed in this augmented space, with the scaling adjusted accordingly. The module is lightweight, parameter‑efficient (due to the low rank \(r\)), and can be initialised from scratch or fine‑tuned. In this paper, we detail the mathematical construction, forward pass, and implementation aspects of TPA, and discuss its advantages and limitations.

---

## 2. Method

### 2.1 Multi‑Head Projections with Tensor Product

Let the input be \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\). Instead of applying a single linear transformation to obtain queries, the module computes two independent projections for queries and keys, each of output dimension \(H \cdot r\), where \(H\) is the number of heads and \(r\) is a hyperparameter (the factor rank). Concretely:

\[
\mathbf{q}_1 = \mathbf{X} \mathbf{W}_{Q1}, \quad \mathbf{q}_2 = \mathbf{X} \mathbf{W}_{Q2},
\]
\[
\mathbf{k}_1 = \mathbf{X} \mathbf{W}_{K1}, \quad \mathbf{k}_2 = \mathbf{X} \mathbf{W}_{K2},
\]

where \(\mathbf{W}_{Q1}, \mathbf{W}_{Q2}, \mathbf{W}_{K1}, \mathbf{W}_{K2} \in \mathbb{R}^{d_{\text{model}} \times (H r)}\). These tensors are reshaped to separate the head dimension: \((B, L, H, r)\). The value projection remains standard:

\[
\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times L \times (H d_h)},
\]

with \(\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (H d_h)}\) and \(d_h = d_{\text{model}} / H\).

### 2.2 Tensor Product Construction of Queries and Keys

For each head \(h\) and each token position, the query vector is formed by the outer product of the two factor vectors:

\[
\mathbf{Q}_{h} = \mathbf{q}_{1,h} \otimes \mathbf{q}_{2,h} \quad \in \mathbb{R}^{B \times L \times H \times r \times r}.
\]

The resulting tensor is then flattened along the last two dimensions to produce a vector of length \(r^2\). After reshaping, the queries and keys have shape \((B, H, L, r^2)\). In the implementation, this is achieved by first computing the outer product via `einsum('blhi,blhj->blhij', q1, q2)` and then reshaping to `(B, L, H, r*r)`.

Similarly, the keys are formed as \(\mathbf{K} = \mathbf{k}_1 \otimes \mathbf{k}_2\), also of shape \((B, H, L, r^2)\). The values are kept in the original per‑head format \((B, H, L, d_h)\).

Thus, the effective head dimension for queries and keys is \(r^2\), while the value dimension remains \(d_h\). This separation allows the attention scores to capture rich pairwise interactions in the factorised space, while the values carry the actual content to be aggregated.

### 2.3 Scaled Dot‑Product Attention

After transpositions to obtain the standard layout \((B, H, L, \cdot)\), the attention scores are computed as:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{r}}.
\]

Note the scaling factor is \(\frac{1}{\sqrt{r}}\) rather than \(\frac{1}{\sqrt{r^2}}\), which may be a design choice to control the variance of the product. A more standard scaling would be \(\frac{1}{r}\), but the code uses \(\frac{1}{\sqrt{r}}\). We'll discuss this in the analysis.

A 2D padding mask (if provided) is expanded and added as an additive bias of \(-10^9\) to mask out invalid positions. Softmax is applied to obtain attention weights \(\mathbf{A} = \operatorname{softmax}(\mathbf{S})\), followed by dropout.

The output context is obtained by the weighted sum of values:

\[
\mathbf{C} = \mathbf{A} \mathbf{V} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

### 2.4 Output Projection

The heads are concatenated to form a tensor of shape \((B, L, H d_h)\) and passed through the output linear layer \(\mathbf{W}_O \in \mathbb{R}^{(H d_h) \times d_{\text{model}}}\) with bias, producing the final output \(\mathbf{Y} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\). The module returns a tuple \((\mathbf{Y}, \text{None})\).

### 2.5 Parameter Count and Complexity

The total number of parameters for the query and key projections is \(4 \times d_{\text{model}} \times H r\) (two for queries, two for keys). For a typical DistilBERT with \(d_{\text{model}}=768, H=12, r=8\), this amounts to \(4 \times 768 \times 12 \times 8 = 294,912\) parameters. The value projection uses \(d_{\text{model}} \times H d_h = 768 \times 768\) (since \(H d_h = d_{\text{model}}\)), i.e., 589,824 parameters. The output projection is also \(d_{\text{model}} \times d_{\text{model}} = 589,824\). Overall, the attention projections are slightly larger than the standard \(4 \times d_{\text{model}}^2\) (≈ 2.36M) for small \(r\), but the query/key projections are much more compact; the total is approximately \(4 \times 768 \times 96 + 2 \times 768^2 = 294,912 + 1,179,648 = 1,474,560\), which is about 62% of the standard size. As \(r\) decreases, the savings increase. For \(r=4\), the query/key part drops to 147,456, total ≈ 1.33M.

The computational complexity of the attention scores is \(O(B H L^2 r^2)\) due to the \(r^2\)-dimensional dot products, compared to \(O(B H L^2 d_h)\) for standard attention. Since \(r^2\) can be larger than \(d_h\) (e.g., \(r=8 \Rightarrow 64\) which equals \(d_h\) for DistilBERT), the dot‑product cost is comparable to the baseline. However, the outer product computation adds an overhead of \(O(B L H r^2)\) per projection. The main novelty is not computational savings but representational richness.

---

## 3. Implementation Details

- **Class `TPAttention`** inherits from `nn.Module` and is registered under the key `"tpa"` in the `ATTENTION_REGISTRY`.
- **Constructor parameters**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rank`: the factor rank \(r\) (default 8).
- **Layers**:
  - `q_proj1`, `q_proj2`: Linear layers from `hidden_size` to `num_heads * rank`.
  - `k_proj1`, `k_proj2`: similarly for keys.
  - `v_proj`: standard Linear to `num_heads * head_dim`.
  - `out_proj`: standard Linear back to `hidden_size`.
  - `dropout` on attention weights.
- **Forward pass**:
  1. Project to \(q_1, q_2\) and reshape to \((B, L, H, r)\).
  2. Compute outer products to get \(Q\) and \(K\) of shape \((B, L, H, r^2)\).
  3. Transpose and reshape to \((B, H, r^2, L)\) and then \((B, H, L, r^2)\) after some transpositions? The code does: `Q = Q.transpose(1,2).transpose(2,3)` leading to \((B, H, r^2, L)\); then for scores, uses `Q.transpose(-2,-1)` to get \((B, H, L, r^2)\) and multiplies with \(K\) of shape \((B, H, r^2, L)\)? Let's carefully trace:
     - After outer product: Q shape `(B, L, H, r*r)`.
     - `Q.transpose(1,2)` → `(B, H, L, r*r)`.
     - `Q.transpose(2,3)` → `(B, H, r*r, L)`.
     - Similarly for K.
     - Scores: `torch.matmul(Q.transpose(-2,-1), K)` → `Q.transpose(-2,-1)` gives `(B, H, L, r*r)`, `K` is `(B, H, r*r, L)`, product yields `(B, H, L, L)`. So the attention matrix is \(L \times L\) as usual, with dot products over the \(r^2\) dimension. The scaling uses `r ** 0.5`.
  4. Apply mask (2D→4D additive).
  5. Softmax, dropout.
  6. Multiply with V (shape `(B, H, L, d_h)`), where V had been transposed to `(B, H, L, d_h)` previously.
  7. Transpose, concatenate heads, output projection.
- **Mask handling**: Assumes 2D mask with 1 for valid, converts to additive `(1-mask)*-1e9`. Other shapes passed through.

---

## 4. Usage and Configuration

Set the attention type to `"tpa"` in the compression config. The rank \(r\) can be provided via `comp_config.rank`. Example:

```python
comp_config.attention_type = "tpa"
comp_config.rank = 12
```

The module does not include pretrained weight transfer; an injection function could be added to initialise the factorised projections from a pretrained model, but currently it is randomly initialised. It can be fine‑tuned from scratch or after copying the original Q/K/V weights (e.g., by aggregating the original Q weight into the two factor projections via SVD). This is left for future integration.

---

## 5. Discussion

TPAttention introduces a novel way to build query and key representations. The tensor product mechanism allows the model to learn multiplicative interactions between two independent low‑rank feature sets, effectively increasing the dimension of the dot‑product space without a proportional increase in parameters. This can lead to richer attention patterns, as each head can attend based on a more complex similarity measure than a simple linear dot product.

However, several aspects require scrutiny:

- **Scaling factor**: The score scaling uses \(\frac{1}{\sqrt{r}}\) rather than \(\frac{1}{\sqrt{r^2}} = \frac{1}{r}\). If the elements of \(q_1, q_2\) are uncorrelated with unit variance, the outer product entries will have unit variance, so the dot product between two such \(r^2\)-dimensional vectors will have variance \(r^2\). To keep the softmax input at a reasonable scale, one should divide by \(r\) (the standard deviation). Using \(\sqrt{r}\) will cause the scores to scale by \(\sqrt{r}\), potentially leading to overly peaked softmax distributions as \(r\) increases. This may be intentional to sharpen attention, but could also cause training instability. A more principled choice would be \(\frac{1}{r}\) or even \(\frac{1}{d_h}\) if the idea is to match the variance of the original attention. This is a parameter that may need tuning.

- **Parameter efficiency vs. capacity**: While the query/key projections are compressed, the outer product effectively creates a large number of features (\(r^2\)) that are entirely determined by the \(2r\) factors. This is a form of low‑rank parameterisation of the query/key space, similar to low‑rank matrix factorisation, but applied in a tensor product fashion. It can capture interactions but may also constrain the representational space.

- **Computational overhead**: The outer product and the \(r^2\)-dimensional dot product add a constant factor overhead. For very small \(r\), this is negligible; for larger \(r\) (e.g., 32 giving 1024), the dot product becomes more expensive than standard attention (where \(d_h=64\)). Thus the method is best suited for small to moderate \(r\).

- **No position encoding enhancement**: The module does not integrate relative position biases or rotary embeddings, which could further improve performance. The tensor product could be naturally combined with such schemes.

- **Initialisation**: Starting from random initialisation for the factor projections may require careful training. An SVD‑based initialisation from a pretrained model’s Q/K weights could be developed to provide a good starting point, as done in other SVD‑based modules.

Despite these concerns, TPA offers a fresh perspective on attention design, encouraging the exploration of structured multiplicative interactions. It can serve as a building block for more complex efficient attention mechanisms.

---

## 6. Conclusion

We have introduced TPAttention, a self‑attention module that constructs queries and keys via the tensor product of two low‑rank factorised features. The design enriches the attention representation while maintaining parameter efficiency. The module integrates seamlessly into the compression framework and can be used as a drop‑in replacement for DistilBERT’s attention sub‑layer. Future work will focus on optimal scaling, pretrained weight initialisation, and combination with positional encoding schemes. TPA highlights the potential of tensor product methods for improving attention expressiveness in Transformers.
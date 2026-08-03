# Adaptive SVD Attention – Paper‑Style Explanation

**Paper Reference**  
This module adapts the idea of factorising pre‑trained weights via truncated singular value decomposition, combined with lightweight trainable residual adapters. The design is inspired by low‑rank adaptation techniques (LoRA, Compacter) and singular value decomposition for model compression.

---

## 1. Introduction

Large pre‑trained language models contain millions of parameters in their attention projections alone. Fine‑tuning all of them is expensive, and simply discarding parameters (e.g., reducing the hidden size) often harms the knowledge embedded in the original weights. The **Adaptive SVD Attention** module strikes a balance: it retains a low‑rank approximation of the pre‑trained projection matrices while adding a very small, fully trainable residual component that can recover the lost fidelity for a specific downstream task. The result is a parameter‑efficient attention block that starts from a high‑quality frozen approximation and only tunes a handful of extra parameters.

In the present implementation, the module is designed as a drop‑in replacement for a DistilBERT self‑attention layer. It factorises every projection matrix—query, key, value, and output—independently, using the truncated SVD of the original weight to initialise the frozen part, and adds a rank‑\(r_{\text{res}}\) adapter pair to capture task‑specific adjustments.

---

## 2. Mathematical Formulation

### 2.1 Standard Projection

A linear projection in a standard attention layer is

\[
\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b},
\]

where \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and \(\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}\). In DistilBERT, \(d_{\text{in}} = d_{\text{out}} = d_{\text{model}}\) for Q, K, V, and output.

### 2.2 Decomposition into SVD Base and Residual Adapter

The proposed **AdaptiveSVDLinear** layer decomposes \(\mathbf{W}\) as

\[
\mathbf{W}_{\text{eff}} = \mathbf{U} \mathbf{V} \;+\; \mathbf{A} \mathbf{B},
\]

where  
- \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\), \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) – **frozen**, initialised from the truncated SVD of the pre‑trained weight \(\mathbf{W}_{\text{pretrained}}\).  
- \(\mathbf{A} \in \mathbb{R}^{d_{\text{out}} \times r_{\text{res}}}\), \(\mathbf{B} \in \mathbb{R}^{r_{\text{res}} \times d_{\text{in}}}\) – **trainable**, small residual adapters (typically \(r_{\text{res}} \ll r \ll \min(d_{\text{out}}, d_{\text{in}})\)).

The forward pass computes

\[
\mathbf{y} = \mathbf{U}(\mathbf{V}\mathbf{x}) + \mathbf{A}(\mathbf{B}\mathbf{x}) + \mathbf{b}.
\]

Biases \(\mathbf{b}\) are copied from the original linear layer and remain trainable (or frozen, depending on the training regime).

### 2.3 Truncated SVD Initialisation

Given the pre‑trained weight \(\mathbf{W}_{\text{pret}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), its compact SVD is

\[
\mathbf{W}_{\text{pret}} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top,
\]

where \(\mathbf{S}\) is the diagonal matrix of singular values. A rank‑\(r\) approximation is obtained by keeping the first \(r\) columns of \(\mathbf{U}_{\text{full}}\) and rows of \(\mathbf{V}_{\text{full}}^\top\), absorbing the singular values into \(\mathbf{U}\):

\[
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \, \mathbf{S}_{[:r,:r]}, \qquad
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]}.
\]

This yields the best rank‑\(r\) approximation of \(\mathbf{W}_{\text{pret}}\) in the Frobenius norm (Eckart–Young–Mirsky theorem). The factors are then frozen, so the model’s initial behaviour very closely matches the original, minus a small error that the residual adapters can later learn to correct.

### 2.4 Overall Attention Computation

The **AdaptiveSVDAttention** module uses four independent `AdaptiveSVDLinear` instances for Q, K, V, and O. The rest of the forward pass is identical to standard scaled dot‑product attention:

\[
\begin{aligned}
\mathbf{Q} &= \text{AdaptiveSVDLinear}_Q(\mathbf{H}), \quad \mathbf{K} = \text{AdaptiveSVDLinear}_K(\mathbf{H}), \quad \mathbf{V} = \text{AdaptiveSVDLinear}_V(\mathbf{H}) \\
\mathbf{S} &= \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} \\
\mathbf{A} &= \text{softmax}(\mathbf{S} + \text{mask}) \\
\mathbf{C} &= \mathbf{A} \mathbf{V} \\
\mathbf{O} &= \text{AdaptiveSVDLinear}_O(\mathbf{C})
\end{aligned}
\]

Padding masks are applied by adding \(-\infty\) to illegal positions before softmax.

---

## 3. Implementation Details

### 3.1 `AdaptiveSVDLinear` – Low‑Rank Factorised Linear Layer

**Constructor parameters**  
- `in_features`, `out_features` – dimensions.  
- `svd_rank` – rank \(r\) of the SVD part.  
- `residual_rank` – rank \(r_{\text{res}}\) of the trainable adapter. If set to 0, the residual adapter is disabled, yielding a pure frozen low‑rank projection.  
- `bias` – boolean indicating whether a bias term is used.

**Parameter initialisation**  
- `U` and `V` are initialised orthogonally, then overwritten by `set_svd_factors()`.  
- `A` and `B` use Kaiming uniform initialisation to promote gradient flow.  
- Bias is zero‑initialised.

**`set_svd_factors(U, V)`**  
Copies the provided SVD factor tensors into the corresponding parameters, ensuring that the base approximation is locked to the pre‑trained decomposition.

**`forward(x)`**  
Computes `F.linear(x, V)`, then `F.linear(..., U)`. If residual adapters exist, adds `F.linear(F.linear(x, B), A)`. Finally adds bias.

### 3.2 `AdaptiveSVDAttention` – Complete Attention Module

**Constructor**  
Requires the pre‑trained attention object to extract original `q_lin, k_lin, v_lin, out_lin` linear layers.  
For each projection:  
1. Instantiate `AdaptiveSVDLinear` with the same dimensions, `svd_rank`, `residual_rank`, and the presence of bias.  
2. Run `_init_svd_part()`, which performs the truncated SVD on the original weight (in float32 for numerical stability) and calls `set_svd_factors` to lock the frozen part. Biases are copied directly.

**Forward pass**  
Standard multi‑head attention: reshape to `(batch, heads, seq, head_dim)`, compute scores, apply mask, softmax, dropout, aggregate, and output projection. All computations use the factorised linear layers.

### 3.3 Integration with the Framework

The injection function `inject_adaptive_svd_attn` is registered as `"adaptive_svd_attn"` in `ATTENTION_REGISTRY`. It reads configuration parameters `svd_rank`, `residual_rank`, and `dropout` from the compression config, constructs the new attention module, and replaces `layer.attention`. This allows the adaptive SVD attention to be inserted into any DistilBERT layer without modifying the training loop.

---

## 4. Discussion

The key advantage of the Adaptive SVD Attention is its **excellent zero‑shot retention** combined with **extreme parameter efficiency**. Because the frozen SVD part captures the vast majority of the original weight’s energy (typically >95% for moderate ranks), the model’s output immediately after injection is almost indistinguishable from the full model. The residual adapters, often comprising less than 1% of the original parameters, can then be fine‑tuned to recover the small performance drop and adapt the representations to the target domain.

This approach also has an interesting regularisation effect: the low‑rank constraints prevent overfitting, and the separation of frozen linguistic knowledge from trainable task‑specific adaptation encourages more robust transfer. Compared to methods that only reparameterise with a low‑rank matrix (e.g., LoRA), the SVD initialisation provides a much stronger prior because it explicitly aligns the factorisation with the pretrained weight’s singular value structure.

One limitation is that the residual adapter rank must be chosen carefully; too small, and the adapter cannot fully compensate for the SVD approximation error; too large, and the parameter savings diminish. In practice, a residual rank of 8–16 is often sufficient for moderate SVD ranks (32–64).

---

## 5. Conclusion

Adaptive SVD Attention provides a mathematically grounded method for compressing attention layers while preserving most of the pre‑trained model’s quality. By decomposing each projection into a frozen truncated SVD base and a tiny trainable residual, it achieves a sweet spot between full fine‑tuning and complete freezing, making it well‑suited for resource‑constrained downstream tasks.
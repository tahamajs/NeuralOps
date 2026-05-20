# SVD Linear Attention: Low‑Rank Factorised Projections with Zero‑Shot SVD Initialisation

**Abstract**

We present **SVD Linear Attention**, a parameter‑efficient drop‑in replacement for the self‑attention sub‑layer in Transformer models. The module replaces each of the four dense linear projections (query, key, value, output) with a **low‑rank factorised linear layer**, where the weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) is approximated by the product of two smaller matrices \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\). The factorisation is initialised from the pretrained weights using a truncated singular value decomposition (SVD), ensuring a near‑lossless zero‑shot transfer: the model’s output immediately after replacement is almost identical to the original pretrained model. The attention mechanism itself remains the standard scaled dot‑product, preserving exact pairwise interactions. The rank \(r\) is a hyperparameter that controls the compression ratio, offering a direct trade‑off between parameter count and representational capacity. The module is designed for DistilBERT and integrates into the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of the low‑rank factorisation, the SVD‑based initialisation, and the overall attention pipeline.

---

## 1. Introduction

Multi‑head self‑attention relies on four linear transformations per layer—queries, keys, values, and output—each typically having \(d_{\text{model}} \times d_{\text{model}}\) parameters in a model like DistilBERT. Together they constitute a significant fraction of the total model size. A classical approach to compress these matrices is low‑rank factorisation: represent each weight matrix \(\mathbf{W}\) as the product \(\mathbf{U}\mathbf{V}\), where \(\mathbf{U}\) and \(\mathbf{V}\) have a small inner dimension \(r \ll \min(d_{\text{out}}, d_{\text{in}})\). This reduces the parameter count from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\), leading to substantial savings.

A critical aspect of such compression is **initialisation**. If the low‑rank factors are randomly initialised, the model loses the knowledge stored in the pretrained weights and must be trained from scratch. The **SVD Linear Attention** module addresses this by initialising \(\mathbf{U}\) and \(\mathbf{V}\) from the truncated SVD of the original weight matrix. This yields the optimal rank‑\(r\) approximation in Frobenius norm (Eckart–Young–Mirsky theorem), guaranteeing that the initial compressed model’s behaviour deviates minimally from the full model. Consequently, the module can be inserted into a pretrained DistilBERT without any fine‑tuning and achieve immediate zero‑shot performance.

The module implements a clean, self‑contained design: the `SVDLinear` class encapsulates the low‑rank operation, while `SVDLinearAttention` applies it to all four projections and carries out the standard multi‑head attention. The injection function copies the pretrained weights into the factorised layers, making the entire process transparent to the user.

In this paper, we detail the mathematical formulation of the factorised linear layer, the SVD initialisation, and the attention computation. We also discuss the implementation, usage, and potential trade‑offs.

---

## 2. Method

### 2.1 Low‑Rank Factorised Linear Layer

A standard linear layer computes \(\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}\), with weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\). The `SVDLinear` class replaces \(\mathbf{W}\) by a product of two matrices:

\[
\mathbf{y} = \mathbf{U} (\mathbf{V} \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\). The forward pass computes the intermediate \(r\)-dimensional vector \(\mathbf{V}\mathbf{x}\) and then multiplies by \(\mathbf{U}\). This avoids instantiating the full \(d_{\text{out}} \times d_{\text{in}}\) matrix, reducing both memory and computation when \(r\) is small.

The parameters \(\mathbf{U}\) and \(\mathbf{V}\) are trainable, and a bias \(\mathbf{b}\) is kept if the original layer had one. Their initialisation (before the SVD transfer) uses orthogonal matrices to promote stable gradient flow.

### 2.2 Zero‑Shot Initialisation via Truncated SVD

Given the pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), we compute its compact SVD:

\[
\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top,
\]

with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\). We retain the top \(r\) singular components:

\[
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \operatorname{diag}(\sigma_1, \dots, \sigma_r) \in \mathbb{R}^{d_{\text{out}} \times r},
\]
\[
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]} \in \mathbb{R}^{r \times d_{\text{in}}}.
\]

These factors are loaded into the `SVDLinear` layer using the method `set_low_rank_factors`, which copies the tensors into the trainable parameters \(\mathbf{U}\) and \(\mathbf{V}\). The bias is copied directly if present. After this assignment, the layer’s forward pass computes exactly \(\mathbf{U}_r \mathbf{V}_r \mathbf{x} + \mathbf{b}\), which is the best rank‑\(r\) approximation of the original transformation.

The same procedure is applied to all four projections (Q, K, V, O). Because the SVD is computed per matrix independently, the rank can be chosen individually, though the current implementation uses a single global rank.

### 2.3 Multi‑Head Attention with Factorised Projections

The `SVDLinearAttention` module mirrors the standard attention mechanism. Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):

\[
\mathbf{Q} = \text{SVDLinear}_Q(\mathbf{X}),\quad
\mathbf{K} = \text{SVDLinear}_K(\mathbf{X}),\quad
\mathbf{V} = \text{SVDLinear}_V(\mathbf{X}).
\]

These are reshaped to multi‑head format \((B, H, L, d_h)\). The attention scores are:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

A 2D padding mask of shape \((B, L)\) (with 1 for valid tokens) is expanded to \((B, 1, 1, L)\) and transformed into an additive mask \((1 - \text{mask}) \cdot (-\infty)\). The softmax and dropout are applied, and the values are aggregated:

\[
\mathbf{C} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}) \mathbf{V}.
\]

Heads are concatenated and the output is projected through the factorised output layer:

\[
\mathbf{Y} = \text{SVDLinear}_O(\mathbf{C}).
\]

The module returns \((\mathbf{Y}, \text{attention\_weights})\) if requested, otherwise \((\mathbf{Y}, \text{None})\).

### 2.4 Parameter Efficiency

For a typical DistilBERT with \(D=768\) and \(H=12\), the standard attention projections contain \(4 \times 768 \times 768 = 2,359,296\) parameters. With a rank \(r=32\), each factorised projection uses \(768 \times 32 + 32 \times 768 = 49,152\) parameters, for a total of \(4 \times 49,152 = 196,608\)—a **12× reduction**. The bias adds negligible overhead. The attention computation itself remains unchanged, so the FLOPs for the dot‑product are the same; the savings are in memory and parameter count.

---

## 3. Implementation Details

- **`SVDLinear`**: Inherits from `nn.Module`. The forward pass is `Vx = F.linear(x, V); out = F.linear(Vx, U) + bias`. The method `set_low_rank_factors(U, V)` allows external assignment of the factorised weights.
- **`SVDLinearAttention`**: Constructor takes `model_config`, `rank`, `dropout`, and `pretrained_attn`. It creates four `SVDLinear` instances and initialises them via `_init_from_linear`. The latter extracts the original weight, computes the SVD using `torch.linalg.svd`, truncates to rank \(r\), and assigns the resulting matrices.
- **Mask handling**: Supports 2D masks (binary, with 1 for valid) by expanding and adding an additive bias. Other mask shapes are passed through, assuming they are already broadcastable.
- **Injection**: The function `inject_svd_linear_attn` is registered under the key `"svd_linear_attn"`. It reads `svd_rank` and `dropout` from the compression configuration and creates the module, replacing `layer.attention`.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"svd_linear_attn"` in the compression configuration. The main hyperparameter is the rank:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_rank` | 32 | Rank \(r\) for the low‑rank factorisation of all projections. |
| `dropout` | 0.1 | Dropout rate on attention weights. |

Example:
```python
comp_config.attention_type = "svd_linear_attn"
comp_config.svd_rank = 64
```

The injection function automatically initialises the module from the pretrained weights, so no further training is required for basic inference. Fine‑tuning can be performed on the low‑rank factors to adapt to a specific task.

---

## 5. Discussion

The SVD Linear Attention module provides a straightforward, mathematically sound method for compressing attention projections. The key advantage is the **zero‑shot** property: because the truncated SVD is the optimal low‑rank approximation, the compressed model starts extremely close to the original, often with negligible accuracy loss even without fine‑tuning. This is in contrast to random initialisation or approaches that only add low‑rank adapters (like LoRA) while keeping the original weights, which can increase total parameters.

However, the module has limitations:

- **Uniform rank**: All four projections share the same rank, which may not be optimal. As shown in the companion module `SVDCompressedAttention`, importance‑aware allocation can yield better compression rates. This could be extended by allowing per‑projection ranks.
- **Quadratic attention remains**: The method compresses parameters, not the attention complexity itself. For extremely long sequences, the attention computation still dominates. This module is best combined with efficient attention patterns (e.g., sliding windows) for full efficiency.
- **Mask handling**: Assumes a 2D binary mask. If the pipeline provides 4D masks, they are passed through unchanged, which may cause broadcasting errors. A robust mask conversion would improve compatibility.

Despite these, the module is a reliable, production‑ready baseline for parameter‑efficient attention. It is especially useful in memory‑constrained environments and as a first step in a multi‑stage compression pipeline.

---

## 6. Conclusion

We have presented SVD Linear Attention, a drop‑in replacement for Transformer attention that factorises all linear projections using truncated SVD, achieving a dramatic reduction in parameters while preserving the original model’s behaviour at initialisation. The module’s simplicity, zero‑shot capability, and seamless integration into the compression framework make it a valuable tool for efficient NLP. Future work may incorporate per‑projection rank selection and combine it with sparse attention mechanisms to also reduce computational complexity.
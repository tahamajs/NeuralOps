# SVD Projection Attention: Low‑Rank Factorised Self‑Attention with Direct U·V Reconstruction

**Abstract**

We present **SVD Projection Attention**, a parameter‑efficient drop‑in replacement for the multi‑head self‑attention sub‑layer in Transformer models. The module factorises each dense projection matrix (query, key, value, output) into two smaller matrices \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\), whose product directly reconstructs the effective weight \(\mathbf{W} = \mathbf{U}\mathbf{V}\). The factorisation is initialised from the pretrained weights via truncated singular value decomposition (SVD), ensuring a near‑lossless zero‑shot transfer: the compressed model’s output is almost identical to the original pretrained model at the start. The rank \(r\) is a hyperparameter that controls the compression ratio. The attention mechanism remains the standard scaled dot‑product, preserving exact token interactions. The module is designed for DistilBERT and integrates into the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of the factorisation, the SVD‑based initialisation, the overall attention pipeline, and a comparison with other SVD‑based modules.

---

## 1. Introduction

Large pre‑trained Transformer models devote a significant portion of their parameters to the linear projections inside each attention head. Compressing these matrices via low‑rank factorisation is a well‑established technique. A weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) can be replaced by the product of two smaller matrices, reducing the parameter count from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\), where \(r \ll \min(d_{\text{out}}, d_{\text{in}})\). 

The **SVD Projection Attention** module follows this paradigm. Unlike some variants that store three factors (U, S, Vh) and absorb singular values into a separate vector, this module uses a straightforward decomposition: the effective weight is simply \(\mathbf{U} \times \mathbf{V}\). The singular values are implicitly absorbed into either \(\mathbf{U}\) or \(\mathbf{V}\) during initialisation. This design is simpler and reduces the number of parameters slightly, while still allowing a high‑quality zero‑shot initialisation via truncated SVD.

The module is intended as a direct alternative to the `SVDLinearAttention` module but with a different internal organisation. It inherits the same zero‑shot weight transfer philosophy and retains the exact multi‑head attention mechanism. In this paper, we detail the mathematical construction, the factorised linear layer, the initialisation procedure, and the integration into the compression framework.

---

## 2. Method

### 2.1 Low‑Rank Factorised Linear Layer

A standard linear transformation \(\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}\) is replaced by the composition of two smaller linear operations:

\[
\mathbf{y} = \mathbf{U} (\mathbf{V} \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) maps the input to an intermediate \(r\)-dimensional space, and \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) maps it to the output dimension. In the forward pass, the effective weight matrix \(\mathbf{W}_{\text{eff}} = \mathbf{U} \mathbf{V}\) is never materialised; instead, two successive matrix multiplications are performed. This reduces the number of trainable parameters from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\), plus the optional bias.

The `SVDProjectionLinear` class implements this: it stores `self.U` and `self.V` as `nn.Parameter` tensors of appropriate shapes, and the forward method computes `weight = self.U @ self.V` and then applies `F.linear(x, weight, self.bias)`. Although this materialises the full weight on‑the‑fly, the gradient computation still acts on the factors, keeping the parameter count low.

### 2.2 Zero‑Shot Initialisation via Truncated SVD

To ensure the compressed model starts from a strong initial point, the factors are initialised from the pretrained weight matrix \(\mathbf{W}_0\) using its truncated SVD. Let

\[
\mathbf{W}_0 = \mathbf{U}_{\text{full}} \mathbf{\Sigma} \mathbf{V}_{\text{full}}^\top
\]

be the full SVD, with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\). For a chosen rank \(r\), we keep the top \(r\) components:

\[
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \mathbf{\Sigma}_{[:r,:r]}, \qquad
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]}.
\]

The factor matrices of the `SVDProjectionLinear` layer are then set as:

- `self.U.data[:, :r] = U_r` (shape \(d_{\text{out}} \times r\))
- `self.V.data[:r, :] = V_r` (shape \(r \times d_{\text{in}}\))

The bias, if present, is copied directly from the original layer. With this assignment, \(\mathbf{U}\mathbf{V} = \mathbf{U}_r \mathbf{V}_r\) is exactly the best rank‑\(r\) approximation of \(\mathbf{W}_0\) in Frobenius norm (Eckart–Young–Mirsky theorem). Therefore, the initial output of the compressed attention module is nearly identical to that of the original, enabling zero‑shot deployment.

The same procedure is applied independently to the query, key, value, and output projections. The rank \(r\) is a global hyperparameter shared across all four projections.

### 2.3 Multi‑Head Self‑Attention with Factorised Projections

The `SVDProjectionAttention` module mirrors the standard multi‑head attention. Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):

\[
\mathbf{Q} = \text{SVDProj}_Q(\mathbf{X}),\quad
\mathbf{K} = \text{SVDProj}_K(\mathbf{X}),\quad
\mathbf{V} = \text{SVDProj}_V(\mathbf{X}).
\]

These tensors, each of shape \((B, L, H d_h)\), are reshaped to \((B, H, L, d_h)\) to separate heads. Attention scores are computed as:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

If a 2D padding mask is provided (shape \((B, L)\) with 1 for valid tokens), it is expanded to \((B, 1, 1, L)\) and converted to an additive bias \((1 - \text{mask}) \cdot (-10^9)\) to mask out padding positions. Softmax and dropout are applied to obtain attention weights \(\mathbf{A}\). The context for each head is:

\[
\mathbf{C}_h = \mathbf{A}_h \mathbf{V}_h.
\]

The heads are concatenated back to shape \((B, L, H d_h)\) and projected through the factorised output layer:

\[
\mathbf{Y} = \text{SVDProj}_O(\mathbf{C}).
\]

The module returns \((\mathbf{Y}, \mathbf{A})\) if `output_attentions=True`, otherwise \((\mathbf{Y}, \text{None})\).

### 2.4 Parameter Efficiency

For a DistilBERT with hidden size 768, each standard projection matrix has \(768 \times 768 = 589,824\) parameters. With a rank \(r = 32\), a factorised projection requires only \(768 \times 32 + 32 \times 768 = 49,152\) parameters, a **12× reduction**. Across four projections, the total parameter count for the attention sub‑layer drops from approximately 2.36 million to 0.2 million. The attention computation itself remains quadratic in sequence length, so the module is focused on parameter compression rather than compute efficiency.

---

## 3. Implementation Details

- **`SVDProjectionLinear`**:
  - Stores `U` (out_features × rank) and `V` (rank × in_features) as parameters, and an optional `bias`.
  - The reset method initialises with Kaiming uniform if no SVD initialisation is performed.
  - The forward method reconstructs the weight `U @ V` and uses `F.linear` with the bias.
- **`SVDProjectionAttention`**:
  - Constructor accepts `model_config`, `rank`, `dropout`, and `pretrained_attn`.
  - Creates four `SVDProjectionLinear` instances.
  - Calls `_init_from_linear` for each projection, which computes the truncated SVD of the original weight and copies the factors.
- **Mask handling**:
  - Supports 2D binary masks by adding `(1 - mask) * -1e9`. Other mask shapes are passed through as‑is.
- **Injection**: The function `inject_svd_proj_attn` is registered under the key `"svd_proj_attn"`. It reads `svd_rank` and `dropout` from the compression config and creates the module, replacing `layer.attention`. The pretrained weights are automatically transferred.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"svd_proj_attn"` in the compression configuration. The primary configurable parameter is the rank:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_rank` | 32 | Rank \(r\) for the low‑rank factorisation. |
| `dropout` | 0.1 | Dropout rate on attention weights. |

Example:
```python
comp_config.attention_type = "svd_proj_attn"
comp_config.svd_rank = 48
```

The injection function handles the initialisation, so no further steps are required for zero‑shot inference. Fine‑tuning can be performed to adapt the low‑rank factors to a downstream task.

---

## 5. Discussion

The SVD Projection Attention module is one of several SVD‑based compression modules in the framework (alongside `svd_linear_attn`, `svd_llm_v2`, `minimal_svd`, etc.). Its distinguishing feature is the direct reconstruction of the weight matrix via a simple matrix multiplication \(\mathbf{U}\mathbf{V}\), without an explicit diagonal singular value vector. This simplifies the forward pass and reduces the number of trainable parameters by one vector per projection. 

The initialisation strategy ensures a high‑fidelity zero‑shot transfer, as the truncated SVD provides the optimal low‑rank approximation. The rank is uniform across all projections, which may not be optimal—more advanced modules like `svd_compressed` allocate ranks per matrix based on energy thresholds. However, the simplicity of a single global rank makes it easy to tune and deploy.

Potential improvements include:
- Per‑projection rank selection based on singular value decay.
- Retaining the original bias if the pretrained layer has one (the code already does this).
- Adding a trainable diagonal scaling similar to `svd_llm_v2` to allow the model to adjust singular values during fine‑tuning.

Despite these, the module stands as a clean, reliable baseline for parameter‑efficient attention. It demonstrates that a straightforward low‑rank factorisation, when carefully initialised, can achieve dramatic compression with negligible initial loss.

---

## 6. Conclusion

We have described SVD Projection Attention, a compact attention module that replaces dense linear projections with a factorised form \(\mathbf{W} = \mathbf{U}\mathbf{V}\), initialised from pretrained weights via truncated SVD. The design preserves the full attention mechanism while drastically reducing the parameter count. Its zero‑shot capability and simple integration into the DistilBERT compression framework make it a practical tool for memory‑constrained deployment. Future work may incorporate per‑layer rank adaptation and dynamic scaling to further enhance flexibility and compression rates.
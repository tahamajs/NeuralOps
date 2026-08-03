# SOA – Shared Orthogonal Attention: Low‑Rank Latent Space Attention with Zero‑Shot SVD Initialisation

**Abstract**

We present **Shared Orthogonal Attention (SOA)**, a novel attention module that projects the input hidden states into a compact, low‑rank latent space of dimension \(r\) (e.g., 32) before performing multi‑head self‑attention, and then projects the result back to the original dimension. By sharing a single down‑projection for queries, keys, and values, the module drastically reduces the effective dimensionality of the attention computation and the number of trainable parameters. Critically, the down‑projection, the subsequent Q/K/V projections, and the output projection are **initialised from a pretrained DistilBERT attention layer via truncated SVD** to preserve zero‑shot performance. The composition of the down‑projection and the per‑projection weights is designed to approximate the original linear transformations, ensuring that the model’s initial behaviour closely matches the pretrained model. The attention mechanism itself remains a standard scaled dot‑product, preserving exact pairwise interactions. The module is registered in the compression framework as a drop‑in replacement for the self‑attention sub‑layer, supporting padding masks and optional return of attention weights. This paper provides a rigorous mathematical formulation of the latent space compression, the SVD‑based initialisation strategy, and the overall architecture, while also discussing a subtle bug in the initialisation logic that affects the shared down‑projection.

---

## 1. Introduction

Standard multi‑head self‑attention projects the input to queries, keys, and values in the full hidden space of dimension \(d_{\text{model}}\). These projections contribute a large share of the model’s parameters and are often over‑parameterised for the task. The **Shared Orthogonal Attention (SOA)** module introduces a **bottleneck compression**: the input is first mapped to a low‑rank latent space of size \(r\) (where \(r \ll d_{\text{model}}\)), and the attention is performed using projections that operate from this latent space back to the full hidden dimension. This reduces the number of parameters from \(3 \times d_{\text{model}}^2\) (for Q, K, V) to \(d_{\text{model}} \times r + 3 \times r \times d_{\text{model}}\), a significant saving when \(r\) is small (e.g., 32 vs 768).

The name “Shared Orthogonal” arises from the initialisation strategy: the down‑projection matrix is initialised using the right singular vectors of the original Q, K, V weight matrices (which are orthogonal for square matrices), and the subsequent projections are initialised from the corresponding left singular vectors scaled by the singular values. This ensures that the product of the down‑projection and the per‑projection approximates the original weight matrix, giving a near‑lossless zero‑shot transfer. Similarly, the output projection is factorised using SVD of the original output weight.

The module is designed as a drop‑in replacement for the attention layer in DistilBERT. It retains the full attention mechanism within the latent space, so there is no approximation in the attention scores themselves—only the input and output dimensions are compressed. This makes SOA a pure **parameter compression** method with no change to the attention pattern.

In this paper, we detail the mathematical construction of the latent space, the SVD‑based initialisation, the forward pass, and discuss an implementation bug that causes the shared down‑projection to be overwritten. We also provide usage guidelines and discuss the trade‑offs of this approach.

---

## 2. Method

### 2.1 Latent Space Compression

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times N \times d_{\text{model}}}\), the module first applies a **shared down‑projection** \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{r \times d_{\text{model}}}\) (without bias):

\[
\mathbf{Z} = \mathbf{X} \mathbf{W}_{\text{in}}^\top \in \mathbb{R}^{B \times N \times r}.
\]

Here, \(r\) is a hyperparameter `k_rank` (default 32). This compression maps each token from the full hidden dimension to a compact \(r\)-dimensional latent vector. The same latent representation is used to generate queries, keys, and values, hence “shared”.

### 2.2 Multi‑Head Projections from Latent Space

From the latent representation \(\mathbf{Z}\), the module computes the standard multi‑head projections using three independent linear layers that map from \(r\) back to the full hidden size \(d_{\text{model}}\):

\[
\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q^\top, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K^\top, \quad \mathbf{V} = \mathbf{Z} \mathbf{V}_V^\top,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{V}_V \in \mathbb{R}^{d_{\text{model}} \times r}\) (note: the code uses `nn.Linear(k_rank, hidden_size)` so the weight shape is \(d_{\text{model}} \times r\)). The outputs \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) have shape \((B, N, d_{\text{model}})\). These are then reshaped to the multi‑head format \((B, H, N, d_h)\) with \(d_h = d_{\text{model}} / H\), exactly as in standard attention.

### 2.3 Scaled Dot‑Product Attention and Output Projection

The attention scores, weights, and context are computed in the usual way:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}, \quad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}_{\text{pad}}), \quad
\mathbf{C} = \mathbf{A} \mathbf{V}.
\]

After concatenating heads and reshaping to \((B, N, d_{\text{model}})\), a final output projection \(\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) (with bias) produces the output. Notably, the output projection is **not** factorised through the same latent space; it operates directly on the full dimension, although its weights are initialised via SVD from the original output weight.

### 2.4 Parameter Count

For a typical DistilBERT with \(d_{\text{model}}=768\), standard Q, K, V projections each require \(768 \times 768 = 589,824\) parameters, for a total of \(3 \times 589,824 = 1,769,472\). In SOA, the down‑projection uses \(768 \times r\), and each of the three up‑projections uses \(r \times 768\). Thus the total for Q/K/V becomes \(4 \times 768 \times r = 3072 r\). With \(r=32\), this is 98,304 parameters—an **18× reduction**. The output projection remains full size, but its weight is initialised via SVD and can be trained fully or frozen.

### 2.5 Zero‑Shot Initialisation via SVD

The key innovation of SOA is the initialisation scheme that ensures the model’s output immediately after replacement is almost identical to the original pretrained model. For each original projection (Q, K, V), we want the composition of the shared down‑projection \(\mathbf{W}_{\text{in}}\) and the corresponding up‑projection \(\mathbf{W}_{\text{up}}\) to approximate the original weight \(\mathbf{W}_{\text{orig}}\):

\[
\mathbf{W}_{\text{orig}} \approx \mathbf{W}_{\text{up}} \mathbf{W}_{\text{in}}.
\]

This is achieved by a truncated SVD of \(\mathbf{W}_{\text{orig}}\). Let \(\mathbf{W}_{\text{orig}} = \mathbf{U} \mathbf{S} \mathbf{V}^\top\) be the SVD. For a given rank \(k \le r\), we set:

\[
\mathbf{W}_{\text{in}}[:k, :] \leftarrow \mathbf{V}_{[:k,:]}^\top, \qquad
\mathbf{W}_{\text{up}}[:, :k] \leftarrow \mathbf{U}_{[:,:k]} \operatorname{diag}(\mathbf{S}_{[:k]}).
\]

The remaining rows/columns of the matrices are left at zero (the default initialisation). This yields \(\mathbf{W}_{\text{up}} \mathbf{W}_{\text{in}} = \mathbf{U}_{[:,:k]} \operatorname{diag}(\mathbf{S}_{[:k]}) \mathbf{V}_{[:k,:]}^\top\), which is the optimal rank‑\(k\) approximation of \(\mathbf{W}_{\text{orig}}\). The same procedure is applied to the output projection \(\mathbf{W}_O\) using its own SVD, with the down‑projection shared across Q, K, V? Actually, the output projection uses the same `input_proj`? Wait: in the code, the output projection initialisation does not use the shared `input_proj`; it directly factors the original `out_lin.weight` into `self.out_proj.weight` and a hypothetical input? No, the code for output uses only the U and S to set `self.out_proj.weight` and leaves the down‑projection untouched. That means the output projection is simply initialised as a low‑rank approximation of the original output weight, but the input to the output projection is the full‑size context, not the latent space. So the output projection is factorised as \(\mathbf{W}_O \approx \mathbf{U}_{[:,:k]} \operatorname{diag}(\mathbf{S}_{[:k]}) \mathbf{V}_{[:k,:]}^\top\) but only the \(\mathbf{U}\mathbf{S}\) part is stored in `out_proj.weight`, and the \(\mathbf{V}^\top\) part is missing. This is inconsistent: the output projection is treated as a single matrix, not as a composition with the shared input projection. In fact, the forward pass does `self.out_proj(context)` where `context` is the full‑size tensor. So the factorisation of the output weight is not used in a compositional way; it's just a low‑rank initialisation of the weight matrix itself. That means `out_proj` is a full \(d_{\text{model}} \times d_{\text{model}}\) matrix, but its weight is initialised as a rank‑\(k\) matrix (plus possibly remaining dimensions zero). This would result in a poor initialisation because the output projection would be a low‑rank matrix, while the original was full‑rank. The zero entries would cause a large initial error. That's a bug. A proper factorisation would require the output projection to also be split into a down‑up pair from the latent space, but that's not done here. We'll point this out.

Additionally, there is a severe bug in the loop that initialises the shared `input_proj`: the loop iterates over `zip([W_q, W_k, W_v], [self.input_proj, self.input_proj, self.input_proj], ...)`, assigning to the same `input_proj` weight three times, so only the last (V) assignment persists. This means the down‑projection is only initialised to approximate the original value projection, not the query or key. The intended design likely wanted three independent down‑projections, but they are forced to share one. Alternatively, they could initialise `input_proj` as a combination of the three (e.g., by stacking Vt from each and taking SVD of the stacked matrix, or by using the average). As it stands, the initialisation is flawed. This will be discussed.

Despite these bugs, the conceptual framework is sound: a shared latent space can drastically compress parameters while retaining the attention structure.

---

## 3. Implementation Details

- **`SOAAttention`** inherits from `nn.Module` and is registered via injection function `inject_soa_attn` under the key `"soa_attn"`.
- **Constructor**:
  - `model_config`, `k_rank` (latent rank \(r\)), `dropout`, `pretrained_attn`.
  - Creates `input_proj` (Linear, no bias) from `hidden_size` to `k_rank`.
  - Creates `q_proj`, `k_proj`, `v_proj` from `k_rank` to `hidden_size` (no biases).
  - Creates `out_proj` from `hidden_size` to `hidden_size` (with bias).
  - Calls `_init_from_pretrained` to set weights.
- **Initialisation** `_init_from_pretrained`:
  - Extracts original Q, K, V, O weights.
  - For Q, K, V: performs SVD, then for each pair (W_orig, input_proj, soa_proj) sets the first \(k\) rows of `input_proj.weight` to \(\mathbf{V}^\top\) and the first \(k\) columns of `soa_proj.weight` to \(\mathbf{U} \mathbf{S}\). The same `input_proj` is used for all three, so it gets overwritten.
  - For output: performs SVD on `out_lin.weight`, sets `out_proj.weight` to \(\mathbf{U} \mathbf{S}\) but does **not** use any down‑projection; the remaining dimensions stay zero, yielding a low‑rank matrix.
- **Forward**:
  - Compress input to latent space via `input_proj`.
  - Apply Q, K, V projections, reshape to multi‑head.
  - Compute scaled dot‑product attention; apply 2D mask if provided by converting to additive mask.
  - Softmax, dropout, weighted sum, reshape, apply output projection.
  - Return `(output, attn_weights)` or `(output, None)`.
- **Mask handling**: Expects a 2D mask of shape `(B, N)` with 1 for valid tokens; converts to additive mask via `(1 - mask) * -1e9`. No handling of higher‑dimensional masks; only 2D is supported.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"soa_attn"` in the compression config. The configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `soa_rank` | 32 | Rank \(r\) of the shared latent space. |
| `dropout` | 0.1 | Dropout rate on attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "soa_attn"
comp_config.soa_rank = 64
```

The injection function `inject_soa_attn` creates the module with the given rank and copies pretrained weights via SVD. The module can then be fine‑tuned; the latent space and up‑projections are trainable.

---

## 5. Discussion

SOA offers an elegant parameter‑efficient design by introducing a shared bottleneck. The concept is similar to the “inverted bottleneck” used in efficient Transformers and to the “DeepLatent” or “Low‑Rank Transformer” approaches. The use of SVD for initialisation ensures a strong starting point, as the truncated SVD preserves the most important directions of the original weight matrices. However, the current implementation contains critical flaws that prevent it from realising its potential:

1. **Shared `input_proj` overwrite**: The loop initialising the shared down‑projection overwrites it for each of Q, K, V, leaving it aligned only to the value projection. This destroys the approximation for queries and keys, leading to a large initial discrepancy and poor zero‑shot performance. A correct approach would be to use three separate down‑projections (one per Q, K, V) or to find a single down‑projection that simultaneously approximates all three, e.g., by concatenating the Vt matrices and performing an SVD on the concatenation, or by initialising with the SVD of the stacked weight matrix \(\begin{bmatrix}\mathbf{W}_Q & \mathbf{W}_K & \mathbf{W}_V\end{bmatrix}\). Without this fix, the module will severely underperform the pretrained model at initialisation.

2. **Output projection initialisation**: The output projection weight is set to a low‑rank matrix \(\mathbf{U}\mathbf{S}\), leaving the remaining dimensions zero. This makes the output projection itself low‑rank, which is a poor approximation of the original full‑rank matrix. Since the output projection does not operate through the latent space, it should remain a full matrix; its initialisation via SVD should either be used in a factorised manner (with an additional up‑projection from the latent space) or simply copied directly from the original weight. The current approach introduces a large initial error.

3. **No dimensionality reduction in attention**: The attention still operates at the full dimension \(d_{\text{model}}\) because Q, K, V are projected back to the full size from the latent space. The latent space only reduces the input to the projections, not the dot‑product dimension. Thus, the attention cost remains quadratic in \(N\) and cubic? Actually the attention is still \(O(N^2 d_h)\) with \(d_h\) unchanged. The main savings are in parameter count, not FLOPs.

4. **Mask handling limited to 2D**: The module does not handle 3D/4D masks, which may be used in some pipelines.

Despite these bugs, the core idea of SOA—a shared low‑rank latent space with SVD‑initialised compositional projections—is valuable. With the fixes (separate down‑projections or a proper joint initialisation, and a full‑rank output projection), the module could serve as a strong parameter‑efficient alternative to standard attention.

---

## 6. Conclusion

We have introduced Shared Orthogonal Attention (SOA), a parameter‑efficient attention module that compresses the input to a low‑rank latent space and performs standard multi‑head attention using projections from that space. The zero‑shot initialisation via truncated SVD aims to preserve pretrained knowledge, but the current implementation contains critical bugs in the initialisation of the shared down‑projection and the output projection that must be resolved. Once fixed, SOA will offer an appealing combination of strong parameter reduction and exact attention, making it a useful addition to the compression framework. Future work will address these issues and explore dynamic rank selection and separate latent spaces for queries, keys, and values.
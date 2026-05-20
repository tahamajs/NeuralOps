# SOA Fused Attention: Low‑Rank Latent Space Attention with Improved Zero‑Shot SVD Initialisation

**Abstract**

We present **SOA Fused Attention**, a refined version of Shared Orthogonal Attention (SOA) that similarly projects the input hidden states into a compact latent space of dimension \(r\) before performing multi‑head self‑attention, but addresses the initialisation flaws of the original SOA by using a **single shared down‑projection initialised from the query weight’s SVD** and then individually setting the up‑projection weights for queries, keys, and values to their respective low‑rank approximations. The result is a more coherent zero‑shot transfer: the composition of the shared down‑projection and each up‑projection approximates the original pretrained weight, preserving the model’s behaviour at initialisation. However, the output projection is still initialised as a low‑rank matrix, which introduces an initial discrepancy. The module retains exact scaled dot‑product attention and is integrated into the DistilBERT compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of the factorised initialisation, the forward pass, and a critical analysis of remaining issues.

---

## 1. Introduction

Standard multi‑head self‑attention projects the input to queries, keys, and values using full‑rank matrices, contributing millions of parameters. The **SOA Fused Attention** module seeks to drastically reduce these parameters by introducing a shared bottleneck: the input is first mapped to a low‑rank latent space of dimension \(r\) (e.g., 32), and the subsequent Q, K, V projections are learned as up‑projections from that latent space. This architecture reduces the total parameter count for Q, K, V from \(3 d_{\text{model}}^2\) to \(d_{\text{model}} r + 3 r d_{\text{model}} = 4 d_{\text{model}} r\), a saving factor of roughly \(d_{\text{model}} / (4r)\) which can exceed 20× for typical dimensions.

The name “Fused” indicates that the down‑projection is shared across all three projections (unlike a version with independent down‑projections). To ensure zero‑shot performance, the weights are initialised using truncated SVD of the pretrained matrices. Unlike the earlier SOA, which incorrectly overwrote the shared down‑projection, SOA Fused initialises the down‑projection once from the query weight’s right singular vectors and then sets each up‑projection from the corresponding left singular vectors scaled by singular values. This yields a consistent low‑rank approximation for each projection. However, the output projection remains improperly initialised, a flaw we will discuss.

In this paper, we detail the mathematics of the latent compression, the SVD‑based initialisation strategy, the forward pass, and the integration into the compression framework. We also highlight the remaining bug in the output projection and suggest remedies.

---

## 2. Method

### 2.1 Latent Space Compression

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), a shared linear down‑projection \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{r \times d_{\text{model}}}\) (without bias) computes a compact representation:

\[
\mathbf{Z} = \mathbf{X} \mathbf{W}_{\text{in}}^\top \in \mathbb{R}^{B \times L \times r}.
\]

The dimension \(r\) is a hyperparameter `k_rank` (default 32). This latent representation is then used to generate queries, keys, and values.

### 2.2 Multi‑Head Projections from Latent Space

Three independent linear layers map the latent vectors back to the full model dimension:

\[
\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q^\top, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K^\top, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V^\top,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times r}\) (note that the linear layer is defined as `nn.Linear(k_rank, hidden_size)`, so the weight shape is \(d_{\text{model}} \times r\), but the operation is effectively \(\mathbf{x} \mathbf{W}^\top\) giving a \(d_{\text{model}}\)‑dimensional output). The resulting tensors are reshaped to multi‑head format \((B, H, L, d_h)\) with \(d_h = d_{\text{model}} / H\).

### 2.3 Scaled Dot‑Product Attention

The attention mechanism is the standard one:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}, \quad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}_{\text{pad}}), \quad
\mathbf{C} = \mathbf{A} \mathbf{V}.
\]

Padding masks are supported: a 2D mask of shape \((B, L)\) with 1 for valid tokens is expanded and converted to an additive mask \((1 - \text{mask}) \cdot (-\infty)\). After attention, the heads are concatenated and reshaped to \((B, L, d_{\text{model}})\).

### 2.4 Output Projection

Finally, a standard linear layer \(\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) (with bias) projects the context to the output.

### 2.5 Zero‑Shot Initialisation via Compositional SVD

The critical design choice is the initialisation of the down‑projection and the up‑projections such that for each original weight \(\mathbf{W}_{\text{orig}}\) (Q, K, V, O), the product \(\mathbf{W}_{\text{up}} \mathbf{W}_{\text{in}}\) approximates \(\mathbf{W}_{\text{orig}}\). The initialisation proceeds as follows:

1. **For the query projection** (and establishing the shared \(\mathbf{W}_{\text{in}}\)):
   - Compute the SVD of the original query weight: \(\mathbf{W}_Q = \mathbf{U}_Q \mathbf{S}_Q \mathbf{V}_Q^\top\).
   - Let \(r_q = \min(r, \text{rank}(\mathbf{W}_Q))\).
   - Set the first \(r_q\) rows of \(\mathbf{W}_{\text{in}}\) to \(\mathbf{V}_Q[:r_q, :]^\top\) (i.e., the top right singular vectors).
   - Set the first \(r_q\) columns of \(\mathbf{W}_Q\) (the up‑projection weight) to \(\mathbf{U}_Q[:,:r_q] \operatorname{diag}(\mathbf{S}_Q[:r_q])\).

   Thus, \(\mathbf{W}_Q \approx \mathbf{W}_{Q,\text{up}} \mathbf{W}_{\text{in}}\) at rank \(r_q\).

2. **For the key and value projections**:
   - Compute the SVD of the original key weight \(\mathbf{W}_K = \mathbf{U}_K \mathbf{S}_K \mathbf{V}_K^\top\) and value weight \(\mathbf{W}_V = \mathbf{U}_V \mathbf{S}_V \mathbf{V}_V^\top\).
   - **Do not** modify \(\mathbf{W}_{\text{in}}\) further; it remains fixed from the query SVD.
   - For key, set \(\mathbf{W}_{K,\text{up}}[:,:r_k] = \mathbf{U}_K[:,:r_k] \operatorname{diag}(\mathbf{S}_K[:r_k])\).
   - For value, set \(\mathbf{W}_{V,\text{up}}[:,:r_v] = \mathbf{U}_V[:,:r_v] \operatorname{diag}(\mathbf{S}_V[:r_v])\).

   Here \(r_k, r_v\) are \(\min(r, \text{rank})\). The products \(\mathbf{W}_{K,\text{up}} \mathbf{W}_{\text{in}}\) and \(\mathbf{W}_{V,\text{up}} \mathbf{W}_{\text{in}}\) will **not** exactly equal the low‑rank approximations of \(\mathbf{W}_K\) and \(\mathbf{W}_V\) because \(\mathbf{W}_{\text{in}}\) is derived from \(\mathbf{V}_Q^\top\), not from \(\mathbf{V}_K^\top\) or \(\mathbf{V}_V^\top\). Thus the approximation quality for keys and values depends on how well the query’s right singular vectors span the key/value spaces. This is a compromise for sharing the down‑projection.

3. **For the output projection**: The original output weight \(\mathbf{W}_O\) is factorised via its own SVD \(\mathbf{W}_O = \mathbf{U}_O \mathbf{S}_O \mathbf{V}_O^\top\). The code then sets the first \(r\) columns of `self.out_proj.weight` to \(\mathbf{U}_O[:,:r] \operatorname{diag}(\mathbf{S}_O[:r])\) and leaves the rest zero. This treats the output projection as a standalone low‑rank matrix, but the input to it is the full‑size context, so the missing \(\mathbf{V}_O^\top\) part (which would correspond to a down‑projection) is absent. As a result, the output projection is a rank‑\(r\) matrix that poorly approximates the full‑rank original weight, leading to a potentially large initial error.

---

## 3. Implementation Details

- **`SOAFusedAttention`** inherits from `nn.Module` and is registered via injection function `inject_soa_fused_attn` under the key `"soa_fused_attn"`.
- **Constructor**:
  - `model_config`, `k_rank` (latent rank \(r\)), `dropout`, `pretrained_attn`.
  - Creates `input_proj` (Linear, no bias) \(d_{\text{model}} \to r\).
  - Creates `q_proj`, `k_proj`, `v_proj` (Linear, no bias) \(r \to d_{\text{model}}\).
  - Creates `out_proj` (Linear, with bias) \(d_{\text{model}} \to d_{\text{model}}\).
  - Calls `_init_from_pretrained`.
- **Initialisation** `_init_from_pretrained`:
  - Iterates over Q, K, V in order; for Q, it also sets `input_proj` (only once).
  - For each, computes SVD, and for the first \(r\) singular components, sets `input_proj.weight[:r, :] = Vt[:r, :]` (only on Q), and sets the corresponding up‑projection weight columns to \(\mathbf{U} \mathbf{S}\).
  - For output, performs SVD and sets `out_proj.weight[:, :r] = U * S`, but does not handle the \(\mathbf{V}^\top\) part.
  - Copies bias if present.
- **Forward**:
  - Compress input to latent space.
  - Project to Q, K, V and reshape.
  - Scaled dot‑product attention with mask handling (2D mask).
  - Softmax, dropout, weighted sum, concatenate heads.
  - Apply output projection (full size).
  - Return `(output, attn_weights)` or `(output, None)`.
- **Mask handling**: Converts a 2D mask to additive mask; no support for 3D/4D masks.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"soa_fused_attn"` in the compression config. Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `soa_rank` | 32 | Rank \(r\) of the shared latent space. |
| `dropout` | 0.1 | Dropout on attention weights. |

Example:
```python
comp_config.attention_type = "soa_fused_attn"
comp_config.soa_rank = 64
```

The injection function creates the module, copying pretrained weights via SVD. The module can then be fine‑tuned; all parameters are trainable.

---

## 5. Discussion

SOA Fused improves upon the original SOA by correctly initialising the shared down‑projection only once and using consistent low‑rank approximations for Q, K, V. However, two significant issues remain:

1. **Output projection as a low‑rank matrix**: The output projection is not factorised through the latent space, yet its weight is initialised as a rank‑\(r\) matrix. This causes a large initial mismatch, as the output projection is expected to operate on the full‑dimensional context. The correct approach would be either (a) to factor the output projection into a down‑up pair using the same latent space (i.e., \( \mathbf{W}_O \approx \mathbf{W}_{O,\text{up}} \mathbf{W}_{O,\text{down}} \) with \(\mathbf{W}_{O,\text{down}}\) mapping from \(d_{\text{model}}\) to \(r\) and \(\mathbf{W}_{O,\text{up}}\) from \(r\) to \(d_{\text{model}}\)), or (b) to initialise the output projection as a full matrix copied directly from the pretrained weight. The current implementation will degrade zero‑shot performance for the output step.

2. **Shared down‑projection mismatch for K and V**: Because the shared \(\mathbf{W}_{\text{in}}\) is set solely from \(\mathbf{V}_Q^\top\), the key and value up‑projections may not approximate their original weights well. If the query’s right singular vectors differ significantly from those of keys/values, the approximation error could be substantial. A more robust approach would be to use three separate down‑projections (one per projection) or to jointly optimise the shared down‑projection to minimise the total error, e.g., by stacking the \(\mathbf{V}^\top\) matrices and taking their SVD.

3. **Mask handling limited**: Only 2D masks are supported; extension to higher‑dimensional masks (e.g., additive 4D masks) would require additional logic.

Despite these issues, the SOA Fused design demonstrates a promising direction: sharing a low‑rank latent space for attention projections can drastically cut parameters, and SVD‑based initialisation provides a strong foundation. The forward pass remains exact, preserving the full attention matrix. The computational cost of attention is unchanged (quadratic in \(L\)), but the parameter savings and potential for reduced storage are significant.

---

## 6. Conclusion

We have introduced SOA Fused Attention, a parameter‑efficient attention module that compresses the input to a shared low‑rank latent space and performs multi‑head attention using up‑projections from that space. The zero‑shot initialisation leverages truncated SVD to approximate the original weights, but the current implementation has flaws in the output projection and the shared down‑projection’s fidelity for keys and values. With appropriate fixes, SOA Fused could serve as a powerful drop‑in replacement for the attention sub‑layer in DistilBERT, offering substantial parameter reduction while maintaining exact attention. Future work will address the output projection factorisation and explore separate latent spaces for each projection to fully realise the potential of this architecture.
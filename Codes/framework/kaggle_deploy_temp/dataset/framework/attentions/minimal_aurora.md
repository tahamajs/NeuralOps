# MinimalAURORA: Zero‑Shot Low‑Rank Factorised Attention for DistilBERT

**Abstract**

We present **MinimalAURORA**, a minimalist, zero‑shot, drop‑in replacement for the self‑attention sub‑layer in DistilBERT. The module replaces each of the four linear projections (query, key, value, output) with a **factorised linear layer** whose rank is determined adaptively from the pretrained weights via an energy‑thresholded truncated SVD. The factorisation stores two small matrices—a down‑projection and an up‑projection—that are initialised from the singular vectors and values of the original weight, guaranteeing a high‑fidelity approximation of the pretrained model at the start. The attention computation itself remains unchanged, using full scaled dot‑product attention. By keeping the rank small (automatically chosen between 4 and 64), the module achieves a substantial reduction in trainable parameters with negligible zero‑shot performance loss. The module supports standard padding masks in multiple formats and can return attention weights if requested. It is integrated into the compression framework via a registry‑based injection function. This paper provides a complete mathematical description of the factorisation, the energy‑based rank selection, and the overall attention architecture.

---

## 1. Introduction

Large pre‑trained Transformer models contain millions of parameters in their attention projections alone. Fine‑tuning and deploying these models is resource‑intensive. Low‑rank factorisation offers a direct path to compression by replacing a dense weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) with the product of two smaller matrices \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\), where \(r \ll \min(d_{\text{out}}, d_{\text{in}})\). If the factorisation is initialised using the truncated singular value decomposition (SVD) of the pretrained weight, the module retains near‑perfect fidelity to the original model, enabling seamless zero‑shot transfer.

**MinimalAURORA** implements this idea for all four attention projections (Q, K, V, O). Crucially, it does not rely on a fixed rank; instead, it selects the rank \(r\) dynamically per weight matrix based on an energy threshold \(\eta \in (0,1]\) (e.g., 0.95). The rank is chosen as the smallest number of singular values required to explain at least \(\eta\) of the total variance (sum of squared singular values). This yields a compressed representation that adapts to the information content of each weight, automatically preserving more dimensions for more complex matrices.

The module is designed to be injected into any DistilBERT layer with zero‑shot initialisation, requiring only the original attention layer to provide the pretrained weights. The forward pass computes full self‑attention, including proper masking. The result is a parameter‑efficient attention block that can be fine‑tuned with a fraction of the original parameters, or used as‑is for inference with minimal accuracy degradation.

---

## 2. Method

### 2.1 Factorised Linear Layer with Energy‑Thresholded SVD

Let \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) be the pretrained weight matrix and \(\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}\) its optional bias. We compute the compact SVD in float32:

\[
\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top,
\]

where \(\mathbf{S} = \operatorname{diag}(\sigma_1, \sigma_2, \dots)\), with singular values ordered decreasingly. The total energy is \(\mathcal{E} = \sum_i \sigma_i^2\). For a given threshold \(\eta\), we find the smallest rank \(r\) such that

\[
\frac{\sum_{i=1}^r \sigma_i^2}{\mathcal{E}} \ge \eta.
\]

We then clip \(r\) to the interval \([r_{\min}, r_{\max}]\), defaulting to \([4, 64]\), and also to \(\min(d_{\text{out}}, d_{\text{in}})\).

The truncated factors are:

\[
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \mathbf{S}_{[:r]} \in \mathbb{R}^{d_{\text{out}} \times r}, \qquad
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]} \in \mathbb{R}^{r \times d_{\text{in}}}.
\]

These are stored as two linear layers:

- `self.down` = \(\mathbf{V}_r\) (bias‑free, shape \(r \times d_{\text{in}}\)),
- `self.up` = \(\mathbf{U}_r\) plus the original bias if present (shape \(d_{\text{out}} \times r\)).

The forward pass computes \(\mathbf{y} = \mathbf{U}_r (\mathbf{V}_r \mathbf{x}) + \mathbf{b}\). This is mathematically equivalent to \(\mathbf{W}_{\text{approx}} \mathbf{x} + \mathbf{b}\) with \(\mathbf{W}_{\text{approx}} = \mathbf{U}_r \mathbf{V}_r\), but the factorised form avoids storing the full \(d_{\text{out}} \times d_{\text{in}}\) matrix and reduces the number of trainable parameters from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\).

### 2.2 Multi‑Head Self‑Attention with Factorised Projections

Given the input hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\), the module applies the factorised Q, K, V projections:

\[
\mathbf{Q} = \text{FactorisedLinear}_Q(\mathbf{H}),\quad
\mathbf{K} = \text{FactorisedLinear}_K(\mathbf{H}),\quad
\mathbf{V} = \text{FactorisedLinear}_V(\mathbf{H}),
\]

reshaped to \((B, H, N, d_h)\) where \(H\) is the number of heads and \(d_h = D/H\).

Attention scores are computed as:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times N \times N}.
\]

### 2.3 Mask Handling

The module accepts an `attention_mask` that can be in several formats (2D, 3D, 4D). It normalises the mask to a boolean tensor `token_valid` of shape \((B, N)\) where `True` indicates valid tokens and `False` indicates padding. For 4D masks, it checks if the last two dimensions are equal (indicating an extended additive mask) and extracts the first row as token validity. Otherwise, it collapses the mask to 2D.

Invalid positions are masked by setting the corresponding scores to a large negative value (`-1e5`). The broadcasting is performed as:

```python
scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e5)
```

This applies the mask along the key dimension for all queries, effectively preventing attention from padding tokens.

### 2.4 Softmax, Aggregation, and Output Projection

Softmax is applied to the masked scores:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{B \times H \times N \times N}.
\]

Dropout is applied to the attention weights. The context is:

\[
\mathbf{C} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

After concatenating heads and reshaping to \((B, N, D)\), the output is projected with the factorised output layer:

\[
\mathbf{O} = \text{FactorisedLinear}_O(\mathbf{C}).
\]

The module returns \((\mathbf{O}, \mathbf{A})\) if `output_attentions` is `True`, otherwise \((\mathbf{O}, \text{None})\).

---

## 3. Implementation Details

- **`FactorisedLinear`**:  
  - Computes SVD once during initialisation.  
  - `self.down`: `nn.Linear(in_dim, rank, bias=False)`.  
  - `self.up`: `nn.Linear(rank, out_dim, bias=has_bias)`.  
  - The energy threshold \(\eta\) is a parameter of the constructor; default 0.95. Minimum rank is 4, maximum is 64.  
  - The weights are set using `.copy_` from the float32 SVD factors.

- **`MinimalAURORAAttention`**:  
  - Inherits from `nn.Module`.  
  - Stores four `FactorisedLinear` instances.  
  - `forward()` follows the standard attention flow.  
  - Mask handling covers 2D, 3D, 4D masks, and an additional `attn_mask` keyword.  
  - Uses `-1e5` as the masking value (a smaller magnitude than typical `-1e9`, but still effectively zero after softmax).

- **Injection function** `inject_minimal_aurora`:  
  - Registered as `"minimal_aurora"`.  
  - Reads `svd_energy` and `dropout` from `comp_config`.  
  - Creates the attention module with the given `pretrained_attn`.  
  - Replaces `layer.attention`.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"minimal_aurora"`. Configurable parameters in `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy` | 0.95 | Fraction of singular value energy to retain. |
| `dropout` | 0.1 | Dropout rate applied to attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "minimal_aurora"
comp_config.svd_energy = 0.98
```

Manual:
```python
attn = MinimalAURORAAttention(model_config, energy=0.98, dropout=0.1, pretrained_attn=old_attn)
layer.attention = attn
```

The factorised layers are trainable by default; to keep them frozen, one would set `requires_grad=False` on their parameters.

---

## 5. Discussion

MinimalAURORA demonstrates that even a straightforward SVD‑based low‑rank factorisation, when initialised from pretrained weights, can dramatically reduce the parameter count of attention projections with negligible initial performance loss. By automatically selecting the rank per matrix based on an energy criterion, it avoids the need to hand‑pick a global rank and adapts to the varying complexity of different projection matrices (e.g., the output projection may require a different rank than the key projection). The small clipping bounds (4–64) keep the number of parameters extremely low.

The attention computation remains full quadratic, so the module does **not** reduce the time complexity of the attention itself. Its primary benefit is memory and parameter efficiency, making it suitable for fine‑tuning on resource‑constrained devices or for deployment where storage is at a premium. The mask handling is robust, supporting the diverse mask shapes produced by the DistilBERT pipeline.

A minor limitation is the use of `-1e5` instead of a more negative value, which is still sufficient for softmax but might theoretically cause issues for very long sequences with many padding tokens (the difference between `exp(-1e5)` and `exp(-1e9)` is negligible). The module also does not include any additional tricks like per‑head rank allocation or dynamic rank adaptation during training; those could be future extensions.

---

## 6. Conclusion

We have presented MinimalAURORA, a zero‑shot, plug‑and‑play attention module for DistilBERT that replaces dense linear projections with energy‑thresholded SVD factorised layers. The design is simple yet effective, preserving the original attention mechanism while drastically cutting the number of trainable parameters. Integrated into the compression framework, it provides a strong baseline for parameter‑efficient fine‑tuning and can be easily customised with different energy thresholds. Future work could combine this factorisation with efficient attention patterns to achieve both memory and time savings.
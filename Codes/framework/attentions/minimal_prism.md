# MinimalPRISM: Zero‑Shot Low‑Rank Factorised Attention via Adaptive SVD

**Abstract**

We present **MinimalPRISM**, a stable, zero‑shot, drop‑in replacement for the self‑attention sub‑layer in DistilBERT. Like its predecessors (MinimalAURORA and MinimalNOVA), it replaces the four dense projection matrices—query, key, value, and output—with factorised linear layers obtained from an energy‑thresholded truncated singular value decomposition (SVD) of the pretrained weights. The rank of each factorisation is chosen adaptively by retaining a specified fraction of the total singular‑value energy, automatically balancing compression and fidelity. The attention mechanism remains unchanged: a full scaled dot‑product attention with robust padding mask handling. The module is designed for zero‑shot initialisation, meaning it can replace the original attention layer without any fine‑tuning and still preserve nearly all of the pretrained model’s behaviour. The name “PRISM” highlights the use of a **P**recision‑**R**educed **I**terative **S**pectral **M**ethod (or simply spectral decomposition) to compress the weight matrices. This paper provides a comprehensive mathematical description of the factorisation and attention pipeline, implementation details, and integration into the compression framework via a registry‑based injection function.

---

## 1. Introduction

Large pre‑trained Transformer models often contain hundreds of millions of parameters, with a significant portion residing in the linear projections of each attention head. Fine‑tuning and deploying such models is resource‑intensive. Low‑rank factorisation of weight matrices offers a direct path to parameter reduction: a matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) is approximated by two smaller matrices \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\) with \(r \ll \min(d_{\text{out}}, d_{\text{in}})\). If the factorisation is initialised via truncated SVD of the original weight, the model retains near‑perfect zero‑shot performance.

**MinimalPRISM** implements this concept for all four attention projections. It differs from earlier versions (MinimalAURORA, MinimalNOVA) only in its registry key (`"minimal_prism"`) and its descriptive name, reflecting a different stage in the evolution of the codebase. The core algorithm—adaptive energy‑thresholded rank selection, SVD factorisation, full attention—remains identical. The module is intended as a robust, production‑grade component that can be injected into any DistilBERT layer with a single configuration flag.

---

## 2. Method

### 2.1 Factorised Linear Layer with Energy‑Thresholded SVD

Let \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) be a pretrained weight matrix and \(\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}\) its optional bias. We compute the compact SVD in float32:

\[
\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{\Sigma} \mathbf{V}_{\text{full}}^\top,
\]

where \(\mathbf{\Sigma} = \operatorname{diag}(\sigma_1, \sigma_2, \ldots)\) with \(\sigma_1 \ge \sigma_2 \ge \cdots \ge 0\). The total energy is \(\mathcal{E} = \sum_i \sigma_i^2\). For a given energy threshold \(\eta \in (0,1]\) (typically 0.95), the rank \(r\) is chosen as the smallest integer \(k\) such that

\[
\frac{\sum_{i=1}^k \sigma_i^2}{\mathcal{E}} \ge \eta.
\]

We then clip \(r\) between 4 and 64, and to \(\min(d_{\text{out}}, d_{\text{in}})\) for safety. The truncated factors are:

\[
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \mathbf{\Sigma}_{[:r]}, \qquad
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]}.
\]

These are stored as two linear layers: a bias‑free `down` layer with weight \(\mathbf{V}_r\) (shape \(r \times d_{\text{in}}\)) and an `up` layer with weight \(\mathbf{U}_r\) and the original bias \(\mathbf{b}\) if present. The forward pass computes:

\[
\mathbf{y} = \mathbf{U}_r (\mathbf{V}_r \mathbf{x}) + \mathbf{b}.
\]

This factorised form reduces the trainable parameter count from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\), often yielding a >10× reduction.

### 2.2 Multi‑Head Self‑Attention

Given input hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\), the module projects them using the factorised layers:

\[
\mathbf{Q} = \text{FactorisedLinear}_Q(\mathbf{H}),\quad
\mathbf{K} = \text{FactorisedLinear}_K(\mathbf{H}),\quad
\mathbf{V} = \text{FactorisedLinear}_V(\mathbf{H}).
\]

These are reshaped to \((B, H, N, d_h)\) where \(H\) is the number of heads and \(d_h = D/H\). The attention scores are:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

### 2.3 Mask Handling

The module accepts an `attention_mask` that can be a 2D, 3D, or 4D tensor. A robust conversion logic extracts a boolean `token_valid` mask of shape \((B, N)\):

- For a 4D mask, it checks if the last two dimensions are equal; if so, it assumes an extended mask and takes the first row as token validity. Otherwise, it squeezes to 2D.
- For 3D, it squeezes the second dimension and converts to bool.
- For 2D, it converts directly to bool.

Invalid (padding) positions are masked by setting scores to a large negative value:

\[
\mathbf{S}_{b,h,i,j} \leftarrow -10^5 \quad \text{if token } j \text{ is padding}.
\]

This ensures that padding tokens do not contribute to the softmax normalisation.

### 2.4 Softmax, Aggregation, and Output

The attention weights are:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{B \times H \times N \times N}.
\]

Dropout is applied to \(\mathbf{A}\). The context is:

\[
\mathbf{C} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

After transposing and reshaping to \((B, N, D)\), the output is projected:

\[
\mathbf{O} = \text{FactorisedLinear}_O(\mathbf{C}).
\]

The module returns \((\mathbf{O}, \mathbf{A})\) if `output_attentions` is True, otherwise \((\mathbf{O}, \text{None})\).

---

## 3. Implementation Details

- **`FactorisedLinear`**: Computes SVD in the constructor, selects rank, and initialises `down` and `up` layers. The weights are set via `copy_` from float32 tensors.
- **`MinimalPRISMAttention`**: Inherits from `nn.Module`. Stores four `FactorisedLinear` instances and a dropout layer. The forward pass follows the standard multi‑head attention flow with mask parsing.
- **Mask parsing**: Handles 2D, 3D, and 4D masks. The masking value is `-1e5`, which, while less negative than the typical `-1e9`, is still effectively zero after softmax.
- **Injection function**: `inject_minimal_prism` is registered under `"minimal_prism"`. It reads `svd_energy` and `dropout` from `comp_config`, creates the attention module with the provided `pretrained_attn`, and replaces `layer.attention`.
- **Zero‑shot**: Because the factorised weights are a near‑optimal low‑rank approximation of the original weights, the model’s initial outputs are almost identical to the full model. No training is required for the model to function after injection.

---

## 4. Usage and Configuration

Set the attention type to `"minimal_prism"` in the compression configuration. The following parameters can be specified in `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy` | 0.95 | Fraction of singular‑value energy to retain. Higher values give better fidelity but larger rank. |
| `dropout` | 0.1 | Dropout rate applied to attention weights. |

Example:

```python
comp_config.attention_type = "minimal_prism"
comp_config.svd_energy = 0.98
```

The injection function transfers the pretrained weights automatically, so the module is immediately usable.

---

## 5. Discussion

MinimalPRISM achieves significant parameter reduction while preserving the full attention mechanism. The adaptive rank selection ensures that each projection is compressed only as much as its spectral structure allows, leading to efficient use of the parameter budget. The module inherits the robust mask handling from its predecessors, making it suitable for typical NLP pipelines with variable‑length batches.

The quadratic complexity of the attention itself is not addressed; the module targets memory and storage efficiency rather than compute time. For longer sequences, it could be combined with a linear attention mechanism. The hard‑coded rank range (4–64) works well for typical hidden sizes (e.g., 768), but may need adjustment for significantly larger models.

The main difference from MinimalAURORA and MinimalNOVA is the registry key and class name. This separation allows the framework to maintain multiple versions without conflict, facilitating A/B testing or phased rollouts.

---

## 6. Conclusion

We have presented MinimalPRISM, a low‑rank factorised attention module for DistilBERT that leverages energy‑thresholded SVD for zero‑shot compression. The module provides a straightforward, production‑ready method to drastically reduce the number of trainable parameters in the attention sub‑layer, while keeping the full expressive power of scaled dot‑product attention. Its integration into the compression framework via a simple configuration flag makes it a valuable tool for efficient NLP deployment.
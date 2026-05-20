# SVD‑LLM V2 Attention: Truncation‑Aware Singular Value Decomposition for Attention Compression

**Abstract**

We present **SVD‑LLM V2 Attention**, a parameter‑efficient self‑attention module that replaces the dense linear projections of the standard multi‑head attention with a low‑rank factorisation based on truncated singular value decomposition (SVD). Each projection matrix (Q, K, V, Output) is stored as three compact factors: \(\mathbf{U}\) (left singular vectors scaled by singular values), \(\mathbf{s}\) (a trainable diagonal scaling), and \(\mathbf{V}^\top\) (right singular vectors). This representation allows a principled zero‑shot initialisation from pretrained weights while drastically reducing the number of trainable parameters. The module is designed to be “truncation‑aware”: an energy threshold parameter controls the effective rank, but in the current implementation the rank is fixed and any remaining capacity is padded with small random values. The attention computation itself remains the full scaled dot‑product, preserving exact pairwise interactions. The module is integrated into the DistilBERT compression framework via a registry‑based injection function, enabling a plug‑and‑play replacement of the standard attention sub‑layer. This paper provides a rigorous mathematical description of the factorisation, the initialisation procedure, the forward pass, and the intended improvements of the “V2” design over its predecessor.

---

## 1. Introduction

Large pre‑trained Transformer models rely heavily on dense linear transformations in their attention layers, which constitute millions of parameters. Low‑rank factorisation via truncated SVD offers a direct path to compression by replacing a weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) with the product of two smaller matrices, reducing the parameter count from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}})\) for a chosen rank \(r\). The original SVD‑LLM (SVD‑based LLM Compression) approach proposed storing the factors \(\mathbf{U}\) and \(\mathbf{V}^\top\) explicitly, and optionally making the singular values trainable. The **SVD‑LLM V2 Attention** module refines this idea by incorporating an explicit trainable singular value vector \(\mathbf{s}\) and an energy‑threshold concept for determining the effective rank, although the current implementation uses a fixed global rank and pads to that rank with random values when the actual SVD rank is smaller.

The primary goals of the V2 design are:
1. To provide a zero‑shot initialisation from pretrained weights, ensuring the compressed model’s initial output closely matches the full model.
2. To allow dynamic, importance‑driven rank selection via an energy threshold, though not yet fully automated in this code.
3. To keep the factors trainable, enabling fine‑tuning to recover any approximation loss and adapt to downstream tasks.

The module is implemented as a drop‑in replacement for DistilBERT’s attention layer, with standard mask handling and dropout. This paper details the mathematical construction, the initialisation algorithm, the forward pass, and the integration into the compression framework. We also discuss the current limitations and how the V2 design aims to overcome them.

---

## 2. Method

### 2.1 Low‑Rank Representation of Projection Weights

For each linear projection originally represented by a weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and optional bias, the SVD‑LLM V2 module stores three parameters:

\[
\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}, \qquad
\mathbf{s} \in \mathbb{R}^{r}, \qquad
\mathbf{V}_h \in \mathbb{R}^{r \times d_{\text{in}}},
\]

where \(r\) is the designated rank. The effective weight matrix is reconstructed on the fly as:

\[
\mathbf{W}_{\text{eff}} = (\mathbf{U} \odot \mathbf{s}) \, \mathbf{V}_h,
\]

where \(\odot\) denotes column‑wise multiplication: each column of \(\mathbf{U}\) is multiplied by the corresponding entry in \(\mathbf{s}\). This is equivalent to \(\mathbf{U} \operatorname{diag}(\mathbf{s}) \mathbf{V}_h\). The original weight is thus approximated by a rank‑\(r\) matrix, but with the singular values separated as a trainable vector. This separation allows the network to adjust the importance of each singular direction during fine‑tuning.

The bias \(\mathbf{b}\) is not factorised and is retained as a separate parameter (or stored implicitly, but the current code does not include bias in the factorised representation; the original bias is discarded. This is a limitation; a complete implementation would keep the original bias or make it trainable).

### 2.2 Zero‑Shot Initialisation from Pretrained Weights

Given the original pretrained weight \(\mathbf{W}_0\), the module computes the full SVD:

\[
\mathbf{W}_0 = \mathbf{U}_{\text{full}} \mathbf{\Sigma} \mathbf{V}_{\text{full}}^\top.
\]

Let \(r_{\text{actual}} = \min(r, \operatorname{rank}(\mathbf{W}_0))\). The truncated factors for the first \(r_{\text{actual}}\) components are:

\[
\mathbf{U}_{r} = \mathbf{U}_{\text{full}[:,:r_{\text{actual}}]} \mathbf{\Sigma}_{[:r_{\text{actual}}]}, \qquad
\mathbf{s}_{r} = \mathbf{1}_{r_{\text{actual}}}, \qquad
\mathbf{V}_{r} = \mathbf{V}_{\text{full}[:r_{\text{actual}},:]}.
\]

Here, the singular values are absorbed into \(\mathbf{U}\), and \(\mathbf{s}\) is initialised to ones, making the initial effective weight exactly the optimal rank‑\(r_{\text{actual}}\) approximation. If \(r_{\text{actual}} < r\) (i.e., the original matrix has lower rank than the designated \(r\)), the remaining dimensions are **padded** with small random values: \(\mathbf{U}\) is extended with random columns of scale 0.01, \(\mathbf{s}\) is extended with values 0.01, and \(\mathbf{V}_h\) is extended with random rows of scale 0.01. This padding provides additional trainable capacity that can capture residual information during fine‑tuning. This mechanism gives the module the ability to potentially expand beyond the optimal SVD rank.

### 2.3 Multi‑Head Attention with Factorised Projections

During the forward pass, the effective weight matrices \(\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o\) are assembled from the stored factors. The input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) are projected:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_q^\top, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_k^\top, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_v^\top,
\]

where each \(\mathbf{W}\) is computed as described. The tensors are then reshaped to multi‑head format \((B, H, L, d_h)\). The rest of the attention is standard:

\[
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}, \quad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}), \quad
\mathbf{C} = \mathbf{A}\mathbf{V},
\]

where \(\mathbf{M}\) is an additive padding mask derived from a 2D binary mask (1 for valid, 0 for padding) by expanding and adding \((1-\text{mask})\cdot(-\infty)\). After concatenating heads, the output is projected:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_o^\top.
\]

Note that no additional bias is added in the projection layers; the original biases are not stored, which may cause a small initial discrepancy but can be compensated by fine‑tuning.

### 2.4 “Truncation‑Aware” Design Intention

The module includes an `energy_threshold` parameter in its constructor, hinting at a dynamic rank selection based on singular value energy. The idea is to automatically determine the rank \(r\) for each projection such that a fraction \(\eta\) (e.g., 0.95) of the total singular value energy is preserved. However, the current implementation does **not** use this threshold for rank selection; the rank is passed externally or defaulted to 64. The padding mechanism instead offers a way to increase the rank beyond the SVD‑determined optimal rank, which is a different concept. A full “V2” would integrate the energy threshold to adaptively set the rank per matrix before initialisation.

---

## 3. Implementation Details

- **`SVDLLMV2Attention`** inherits from `BaseAttention` and is registered under `"svd_llm_v2"`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rank`: the target rank \(r\) (default 64).
  - `energy_threshold`: unused in current logic.
  - `pretrained_attn`: the original attention module; if provided, calls `_init_from_pretrained`, else `_init_random`.
- **Factor storage**: The parameters are `query_u`, `query_s`, `query_vh`, and similarly for key, value, and output. Their shapes are \((d_{\text{all\_head}}, r)\), \((r,)\), \((r, d_{\text{model}})\).
- **`_init_from_pretrained`**:
  - For each projection, computes full SVD using `torch.linalg.svd` on the original weight.
  - Sets the first \(r_{\text{actual}}\) columns of `u` to \(\mathbf{U}_{[:,\text{:}r]} \mathbf{\Sigma}_{[:r]}\), `s` to ones, and `vh` to \(\mathbf{V}_{[:r,:]}\).
  - If \(r_{\text{actual}} < r\), calls `_pad_factors` to fill the remaining dimensions with small random tensors.
- **`_pad_factors`**: Extends the factor tensors to reach the desired rank by concatenating random normal values scaled by 0.01 for `u` and `vh`, and 0.01 for `s`.
- **Forward pass**:
  - Reconstructs weights: `W = (u * s) @ vh`.
  - Projects hidden states to Q, K, V via `F.linear`.
  - Reshapes to multi‑head, computes scaled dot‑product attention, applies mask, softmax, dropout.
  - Aggregates values, concatenates heads, applies output weight.
  - Returns `(output, None)`.
- **Mask handling**: Converts a 2D binary mask (1 for valid) to additive `(1-mask) * -1e9`. Other mask shapes are passed through.
- **Injection**: `inject_svd_llm_v2` reads `rank` and `svd_energy` from `comp_config`, creates the module, and replaces the layer’s attention. This injection is registered under the same key `"svd_llm_v2"`.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"svd_llm_v2"` in the compression configuration. The main configurable parameters (via `comp_config`) are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 64 | Target rank \(r\) for the factorisation. |
| `svd_energy` | 0.95 | Energy threshold (currently unused). |
| `dropout` | 0.1 | Dropout rate on attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "svd_llm_v2"
comp_config.rank = 32
```

The injection function initialises the factors from the pretrained weights, so the module is immediately functional. Fine‑tuning can be performed to improve task‑specific performance.

---

## 5. Discussion

The SVD‑LLM V2 module offers a flexible and parameter‑efficient alternative to standard attention projections. By factoring out the singular values into a trainable vector \(\mathbf{s}\), the model can learn to rescale the importance of different singular directions, potentially compensating for the truncation error. The padding strategy allows the model to increase its representational capacity beyond the SVD‑derived rank, which is an interesting twist: instead of strictly limiting the rank to preserve energy, it provides additional random dimensions that can be shaped during training.

However, several aspects of the current implementation are suboptimal:

1. **Missing bias**: The original projection biases are discarded. In many pretrained models, biases are small but not zero; discarding them introduces a small initial shift. A full implementation should retain the biases as part of the layer.

2. **Unused energy threshold**: The `energy_threshold` parameter is documented but never used to determine the rank. The rank must be set manually, which undermines the “truncation‑aware” label. A true energy‑thresholded selection would automatically pick the rank per matrix to satisfy a given energy criterion, as in the `MinimalSVDAttention` module.

3. **Padding random values**: Padding with random factors adds trainable parameters that are initially uncorrelated with the original weight. While they can be trained, they start from a non‑optimal state and may require careful learning rate scheduling to avoid disrupting the good SVD initialisation. It might be preferable to initialise the extra dimensions with zeros and let the gradient flow expand them.

4. **Uniform rank**: All projections share the same rank. The method would be more effective if each projection could have its own rank, as the singular spectra of Q, K, V, O matrices may differ.

Despite these issues, the module represents a step forward from the basic SVD‑factorised attention by incorporating trainable singular values and a mechanism to expand capacity. It can serve as a foundation for a more advanced, energy‑aware compression algorithm.

---

## 6. Conclusion

We have described SVD‑LLM V2 Attention, a low‑rank factorised attention module that stores projection weights as \(\mathbf{U}, \mathbf{s}, \mathbf{V}^\top\) and initialises them from a pretrained model via truncated SVD. The trainable singular value vector and the padding mechanism provide additional flexibility. While the current implementation lacks the dynamic rank selection and bias handling, it illustrates the core principles of truncation‑aware compression. The module integrates smoothly into the DistilBERT compression framework and can be used as a building block for further improvements in parameter‑efficient Transformer design. Future work will implement automatic energy‑based rank selection, retain biases, and explore per‑projection rank allocation.
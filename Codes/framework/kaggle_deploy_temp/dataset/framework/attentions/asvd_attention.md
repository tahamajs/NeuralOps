# ASVD Attention: Activation‑Aware Singular Value Decomposition for Attention Compression

**Abstract**

We present a low‑rank compressed attention module, **ASVD Attention**, that replaces the dense linear projections of standard multi‑head self‑attention with truncated Singular Value Decomposition (SVD) approximations. The decomposition separates each projection matrix into three compact factors: \(\mathbf{U}\), \(\mathbf{s}\), and \(\mathbf{V}^\top\), which are stored explicitly and multiplied on‑the‑fly. This reduces the parameter count from \(O(d^2)\) to \(O(rd)\), where \(r \ll d\) is the chosen rank. The module is designed to be initialised directly from a pretrained attention layer via SVD of the original weights, ensuring a strong zero‑shot starting point. The name “Activation‑Aware” reflects the intended future extension where the SVD is weighted by activation statistics to better preserve important directions, but the current implementation uses standard, unweighted SVD. The module integrates seamlessly into existing Transformer frameworks as a drop‑in replacement.

---

## 1. Introduction

Transformer models rely heavily on dense linear transformations in their attention layers. For a hidden size \(d_{\text{model}}\) and \(H\) heads, each projection matrix has shape \(d_{\text{model}} \times (H d_h)\) or \((H d_h) \times d_{\text{model}}\), where \(d_h = d_{\text{model}} / H\). These matrices contain millions of parameters, making fine‑tuning and deployment costly. Low‑rank factorisation offers a direct path to compression: a weight matrix \(\mathbf{W} \in \mathbb{R}^{m \times n}\) can be approximated by \(\mathbf{U} \operatorname{diag}(\mathbf{s}) \mathbf{V}^\top\) with \(\mathbf{U} \in \mathbb{R}^{m \times r}\), \(\mathbf{s} \in \mathbb{R}^r\), \(\mathbf{V}^\top \in \mathbb{R}^{r \times n}\), and rank \(r \ll \min(m,n)\).

The **ASVD Attention** module implements this idea for all four attention projections (query, key, value, output). The factors are stored as trainable parameters and multiplied at each forward pass to reconstruct the full weight, after which standard scaled dot‑product attention proceeds. The initial factors are obtained by computing the SVD of the pretrained weights, yielding an excellent approximation of the original layer with far fewer parameters.

The “activation‑aware” label originates from the concept of weighting the reconstruction error by the variance of the activations, so that directions with larger activation magnitude are preserved more faithfully. While the current code contains a placeholder (`apply_activation_scales`), the SVD used is plain, not weighted. The module thus serves as a baseline for future integration of activation‑aware decomposition.

---

## 2. Method

### 2.1 Low‑Rank Decomposition via SVD

Let \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) be a pretrained weight matrix. The compact SVD is

\[
\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{\Sigma} \mathbf{V}_{\text{full}}^\top,
\]

with singular values \(\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p\). A rank‑\(r\) approximation is given by

\[
\mathbf{W} \approx \mathbf{U}_r \operatorname{diag}(\mathbf{s}_r) \mathbf{V}_r^\top,
\]

where \(\mathbf{U}_r = \mathbf{U}_{\text{full}}[:,:r]\), \(\mathbf{s}_r = [\sigma_1, \dots, \sigma_r]^\top\), and \(\mathbf{V}_r^\top = \mathbf{V}_{\text{full}}[:r, :]\). This is the optimal rank‑\(r\) approximation in Frobenius norm (Eckart–Young–Mirsky theorem).

In the ASVD module, each projection is represented by three parameters:
- \(\mathbf{U}\) of shape \((d_{\text{out}}, r)\),
- \(\mathbf{s}\) of shape \((r,)\),
- \(\mathbf{V}_h\) (the transposed right singular vectors) of shape \((r, d_{\text{in}})\).

During a forward pass, the effective weight is computed as:

\[
\mathbf{W}_{\text{eff}} = (\mathbf{U} \odot \mathbf{s}) \, \mathbf{V}_h,
\]

where \(\odot\) denotes broadcasting element‑wise multiplication of \(\mathbf{s}\) along the columns of \(\mathbf{U}\). This is equivalent to \(\mathbf{U} \operatorname{diag}(\mathbf{s}) \mathbf{V}_h\).

### 2.2 Activation‑Aware Weighting (Conceptual)

In standard SVD, all input dimensions are treated equally. However, in neural networks, certain input channels exhibit much higher variance (activation magnitude) than others. An **activation‑aware SVD** would scale the rows of \(\mathbf{W}\) (or columns) by the square root of the average squared activation, effectively putting more emphasis on recovering the weight structure for highly active dimensions. The objective becomes:

\[
\min_{\text{rank}(\tilde{\mathbf{W}}) \le r} \| (\mathbf{W} - \tilde{\mathbf{W}}) \mathbf{D} \|_F^2,
\]

where \(\mathbf{D}\) is a diagonal matrix containing activation scales. This can be solved by performing SVD on \(\mathbf{W} \mathbf{D}\) and then undoing the scaling. The current `ASVDAttention` class provides a hook `apply_activation_scales(scales)` that is currently a no‑op, but it can be extended to incorporate such scaling before or during the SVD initialisation.

### 2.3 Attention Computation

Given a hidden state tensor \(\mathbf{H} \in \mathbb{R}^{B \times N \times d_{\text{model}}}\):

1. **Projections:** Reconstruct \(\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o\) using the stored factors, then compute  
   \(\mathbf{Q} = \mathbf{H} \mathbf{W}_q^\top\), \(\mathbf{K} = \mathbf{H} \mathbf{W}_k^\top\), \(\mathbf{V} = \mathbf{H} \mathbf{W}_v^\top\).  
   These are reshaped to \((B, H, N, d_h)\).

2. **Scaled Dot‑Product Attention:**  
   \(\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}\),  
   \(\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \text{mask})\),  
   \(\mathbf{C} = \mathbf{A} \mathbf{V}\).

3. **Output Projection:**  
   \(\mathbf{O} = \mathbf{C} \, \mathbf{W}_o^\top\), where \(\mathbf{C}\) is first reshaped to \((B, N, H d_h)\).

The mask is expected to be an additive tensor (0 for valid, \(-\infty\) for invalid) and is broadcast appropriately. Dropout is applied to the attention probabilities.

---

## 3. Implementation Details

### 3.1 Parameter Storage and Forward Pass

The module stores 12 parameter tensors (plus optional dropout):
- `query_u`, `query_s`, `query_vh`
- `key_u`, `key_s`, `key_vh`
- `value_u`, `value_s`, `value_vh`
- `out_proj_u`, `out_proj_s`, `out_proj_vh`

All are `nn.Parameter` and therefore trainable. The forward pass reconstructs the weight matrices on‑the‑fly using

```python
W_q = (self.query_u * self.query_s) @ self.query_vh
```

and similarly for the other projections. Because the matrix multiplication is performed each forward step, there is no memory for the full weight matrices, and gradient updates flow through the factors.

The `_reshape` helper rearranges the tensor to the multi‑head layout.

### 3.2 Initialisation from Pretrained Weights

The static method `from_module(module, rank, activation_scales=None)` is provided to initialise an `ASVDAttention` from an existing attention module (typically from a standard Transformer like those in DistilBERT or BERT). It assumes the original module has a combined QKV projection `in_proj_weight` of shape \((3 \times H d_h, d_{\text{model}})\) and an output projection `out_proj.weight` of shape \((d_{\text{model}}, H d_h)\). The combined weight is split into three equal parts for Q, K, V.

For each split, the SVD is computed in float64 (NumPy), then truncated to the given rank, and the resulting \(U_r\), \(S_r\), \(Vh_r\) are loaded into the corresponding parameters. The output projection is treated similarly.

This yields a zero‑shot compressed model that closely approximates the original.

### 3.3 Integration with the Framework

The class is registered under the name `"asvd"` in the `ATTENTION_REGISTRY`. It can be used by specifying the appropriate configuration (not shown, but typically `rank` is provided via `comp_config` or directly). The module is a subclass of `BaseAttention`, which presumably provides standard hooks for injection.

---

## 4. Discussion

The ASVD attention module offers a simple and interpretable compression method. By keeping the singular values and vectors as separate trainable parameters, the model can fine‑tune the low‑rank approximation end‑to‑end, potentially adapting the subspace to the downstream task. The parameter savings are substantial: for a typical DistilBERT with \(d_{\text{model}}=768\), \(H=12\), the Q, K, V, and O projections each have \(768 \times 768 = 589,824\) parameters. With a rank \(r=64\), each factorised projection requires \(768 \times 64 + 64 + 64 \times 768 = 98,368\) parameters—a \(6\times\) reduction. The combined effect across all layers can reduce the total attention parameters by over 80%.

The current implementation does not exploit activation statistics, but the placeholder indicates that the author intended to incorporate activation‑aware weighting. This is a well‑motivated extension: standard SVD minimises reconstruction error uniformly, but a small error in a frequently activated direction may have a larger impact on the model’s output. By scaling the weight matrix by activation standard deviations before SVD, one can focus the approximation on the most critical directions. This would require collecting activation statistics on a calibration dataset prior to compression.

One limitation is that the reconstructed weight is computed each forward pass with a matrix multiplication, which adds a small computational overhead compared to a plain `nn.Linear`. However, the product `(U*s) @ Vh` is of size \(d_{\text{out}} \times d_{\text{in}}\) and could be pre‑computed if the factors are frozen; the current design leaves all factors trainable, so the weight is dynamic.

---

## 5. Conclusion

ASVD Attention provides a straightforward yet effective means of compressing the attention layers of a Transformer. By replacing dense projections with truncated SVD factors that are fine‑tuned, it achieves significant parameter reduction while retaining a strong initialisation from the pretrained weights. The modular design and the `from_module` initialiser make it easy to drop into existing models. Future work can realise the full “activation‑aware” aspect by incorporating activation scales into the SVD initialisation, further improving the compression–performance trade‑off.
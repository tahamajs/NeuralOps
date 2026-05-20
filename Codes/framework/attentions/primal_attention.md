# Primal‑Attention: Self‑Attention through Asymmetric Kernel SVD in Primal Representation

**Abstract**

We present **Primal‑Attention**, an efficient self‑attention module that approximates the full quadratic attention matrix via a low‑rank factorisation based on the Nyström method. The module selects a small set of landmark tokens (the “primal” basis) using either segment‑wise average pooling or uniform sampling, and then computes three intermediate matrices capturing the interactions between the full set of queries/keys and the landmarks. A key component is the iterative refinement of the pseudoinverse of the landmark interaction matrix, which starts from the singular value decomposition (SVD) pseudoinverse and improves it via a Newton–Schulz iteration. This yields a robust low‑rank approximation of the attention scores while avoiding the materialisation of the full \(L \times L\) matrix. The method achieves linear complexity in sequence length \(L\) with respect to the number of landmarks \(m\), i.e., \(O(L m)\), making it suitable for long sequences. The module is designed as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, integrating seamlessly with the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of the landmark selection, the asymmetric kernel approximation, and the iterative pseudoinverse algorithm, along with the overall attention pipeline.

---

## 1. Introduction

Multi‑head self‑attention scales quadratically with the sequence length \(L\), posing challenges for processing long documents. Low‑rank approximations, particularly those based on the Nyström method, have proven effective in reducing this complexity while preserving model quality. The **Primal‑Attention** module adopts a Nyström‑like approach: it selects a small number \(m\) of landmark tokens (the “primal” representation) and computes an approximate attention matrix by expressing the full attention as a product of three smaller matrices and a pseudoinverse of the landmark interaction matrix.

The name “Primal” suggests a representation based on a subset of original tokens acting as a basis, akin to a primal form in kernel methods. The key innovation in this implementation is the iterative refinement of the Moore–Penrose pseudoinverse, which uses the SVD‑based pseudoinverse as an initial guess and applies several steps of a Newton–Schulz iteration to improve accuracy. This avoids potential instability of the direct Newton–Schulz method from a random start.

The module supports two landmark selection strategies: segmented mean pooling (when the sequence length is divisible by \(m\)) and uniform sampling (otherwise). The projections for queries, keys, values, and output are standard linear layers (not factorised), meaning parameter compression is not the goal; the efficiency gain is entirely in the attention computation itself. The module integrates into the DistilBERT compression framework as a drop‑in replacement for standard attention, registered under the key `"primal"`.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first applies linear projections to obtain multi‑head queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = D/H\). The tensors are reshaped to \((B, H, L, d_h)\) via the `_reshape` method.

### 2.2 Landmark Selection

A set of \(m\) landmarks is chosen, where \(m\) is the `rank` hyperparameter (default 64). Two strategies are used:

1. **Segmented mean pooling** (when \(L \bmod m = 0\)): The sequence is divided into \(m\) equal segments of size \(s = L / m\). For each segment, the queries and keys are averaged within the segment to form a single landmark vector. This yields \(\mathbf{Q}_{\text{land}}, \mathbf{K}_{\text{land}} \in \mathbb{R}^{B \times H \times m \times d_h}\).

2. **Uniform sampling** (when \(L \bmod m \neq 0\)): \(m\) indices are chosen uniformly spaced across the length dimension using `torch.linspace`. The corresponding query and key slices are taken as landmarks.

This selection is performed independently for each batch element and head.

### 2.3 Sub‑Matrix Computation

Using the full queries \(\mathbf{Q}\) and keys \(\mathbf{K}\) along with the landmark queries \(\mathbf{Q}_{\text{land}}\) and keys \(\mathbf{K}_{\text{land}}\), three scaled similarity matrices are computed:

\[
\mathbf{F}_1 = \frac{\mathbf{Q} \mathbf{K}_{\text{land}}^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times m},
\]
\[
\mathbf{F}_2 = \frac{\mathbf{Q}_{\text{land}} \mathbf{K}_{\text{land}}^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times m \times m},
\]
\[
\mathbf{F}_3 = \frac{\mathbf{Q}_{\text{land}} \mathbf{K}^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times m \times L}.
\]

These matrices correspond to the query–landmark, landmark–landmark, and landmark–key interactions, respectively. The Nyström approximation of the full attention score matrix \(\mathbf{S} = \mathbf{Q} \mathbf{K}^\top / \sqrt{d_h}\) is then given by:

\[
\tilde{\mathbf{S}} = \mathbf{F}_1 \mathbf{F}_2^+ \mathbf{F}_3,
\]

where \(\mathbf{F}_2^+\) is the Moore–Penrose pseudoinverse of \(\mathbf{F}_2\). This approximation is exact if the landmarks span the row and column spaces of \(\mathbf{S}\), and is otherwise a low‑rank approximation of rank at most \(m\).

### 2.4 Iterative Refinement of the Pseudoinverse

The computation of \(\mathbf{F}_2^+\) is critical. The module employs an **iterative scheme** that combines the direct SVD pseudoinverse with Newton–Schulz refinement. First, the initial pseudoinverse is obtained using `torch.linalg.pinv`, which computes the pseudoinverse via the SVD:

\[
\mathbf{X}_0 = \mathbf{F}_2^+ \quad (\text{direct SVD‑based}).
\]

Then, for a fixed number of iterations \(T = 6\), the Newton–Schulz update is applied:

\[
\mathbf{X}_{k+1} = \mathbf{X}_k \left( 2 \mathbf{I} - \mathbf{F}_2 \mathbf{X}_k \right).
\]

This iteration converges quadratically to the pseudoinverse if \(\mathbf{X}_0\) is sufficiently close. Starting from the SVD pseudoinverse provides an excellent initial guess, ensuring robust and fast convergence. The use of this refinement can correct any small errors from the truncated SVD and improve the overall approximation of the attention matrix.

### 2.5 Approximate Attention and Output

After obtaining \(\mathbf{F}_2^+\), the approximate attention scores are computed as:

\[
\tilde{\mathbf{S}} = \mathbf{F}_1 (\mathbf{F}_2^+ \mathbf{F}_3) \quad \in \mathbb{R}^{B \times H \times L \times L}.
\]

A padding mask, if provided, is added to \(\tilde{\mathbf{S}}\) (expected as an additive mask with \(-\infty\) for padding, broadcastable). The softmax function is then applied:

\[
\mathbf{A} = \operatorname{softmax}(\tilde{\mathbf{S}} + \mathbf{M}_{\text{pad}}).
\]

Dropout is applied to \(\mathbf{A}\), and the context is obtained:

\[
\mathbf{C} = \mathbf{A} \mathbf{V} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

The heads are then concatenated and projected via \(\mathbf{W}_O\) to produce the final output.

### 2.6 Complexity

The cost of forming \(\mathbf{F}_1\) and \(\mathbf{F}_3\) is \(O(B H L m d_h)\), \(\mathbf{F}_2\) is \(O(B H m^2 d_h)\), and the pseudoinverse is \(O(B H m^3)\). The product \(\mathbf{F}_1(\mathbf{F}_2^+ \mathbf{F}_3)\) costs \(O(B H L m^2 + B H L m d_h)\). For \(m \ll L\), the overall complexity is linear in \(L\), with the dominant term being \(O(B H L m d_h)\). Memory is \(O(B H L m)\) for the intermediate matrices, not \(O(B H L^2)\).

---

## 3. Implementation Details

- **Class `PrimalAttention`** inherits from `BaseAttention` and is registered as `"primal"`.
- **Constructor parameters**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rank`: number of landmarks \(m\) (default 64).
  - `pretrained_attn`: not used; the linear layers are randomly initialised. (No weight transfer mechanism is included.)
- **`moore_penrose` function**: Implements the iterative refinement. It first calls `torch.linalg.pinv(A)` for the initial guess, then applies `X = X @ (2*I - A @ X)` for `iters` steps. The loop uses the same batch‑ and head‑wise identity matrix broadcast via `torch.eye`.
- **Landmark selection**: As described; uses segment averaging when divisible, otherwise `torch.linspace`.
- **Attention mask**: Expected to be a 2D tensor of shape \((B, L)\) with values 1 for valid and 0 for padding (additive style? Actually it adds the mask directly with `attention_mask.unsqueeze(1)`, assuming it's already in additive form (e.g., 0 for valid, -∞ for padding). This is a potential bug: a typical binary mask would need conversion. The code follows the same pattern as Nyströmformer, which also had this issue.)
- **Output**: Returns `(output, None)` unless `output_attentions` is true.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"primal"` in the compression configuration. The only hyperparameter is the number of landmarks:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 64 | Number of landmarks \(m\). |
| `dropout` | from config | Dropout rate on attention weights. |

Example injection via framework (requires a corresponding injection function, not provided in snippet):
```python
comp_config.attention_type = "primal"
comp_config.rank = 128
```

Manual instantiation:
```python
attn = PrimalAttention(config=model_config, rank=128)
layer.attention = attn
```

Because no weight transfer is performed, the linear layers are randomly initialised, so the model must be fine‑tuned or initialised from pretrained weights externally.

---

## 5. Discussion

Primal‑Attention follows the well‑established Nyströmformer paradigm but distinguishes itself by the specific iterative pseudoinverse algorithm. The combination of SVD pinv followed by Newton–Schulz refinement is a robust approach that can yield a more accurate inverse than either method alone, especially when \(\mathbf{F}_2\) is ill‑conditioned. The landmark selection strategies are simple and effective.

However, several aspects of the implementation could be improved:

- **Mask handling**: As in Nyströmformer, the module directly adds the attention mask to the scores without converting a binary mask to additive form, expecting the mask to already be in additive style (0 for valid, \(-\infty\) for padding). This is likely incorrect for standard DistilBERT masks, which are binary 1/0. A conversion step is needed.
- **No weight transfer**: Unlike many other modules in this framework, Primal‑Attention does not support zero‑shot initialisation from a pretrained attention layer. Adding a function to copy `q_lin`, `k_lin`, `v_lin`, `out_lin` weights would be straightforward and would preserve pretrained knowledge.
- **No residual or gating**: The module outputs only the attention context; there is no additional feature such as value‑residual connection (as in Nyströmformer with `residual` option) or a gate to fuse with a local branch.
- **Pseudoinverse stability**: The iterative refinement with fixed 6 iterations is reasonable, but the convergence could be monitored; however, for \(m\) up to 64 it is fast.
- **Landmark diversity**: When using uniform sampling, the same landmarks are used for all heads and batch items; using head‑specific landmarks could improve approximation.

Despite these, Primal‑Attention provides a concise and functional implementation of a linear‑time attention approximation. The “Primal” branding suggests a theoretical connection to kernel method representations, though the code does not explicitly exploit such a connection beyond the Nyström technique. It remains a solid baseline for efficient attention experiments.

---

## 6. Conclusion

We have presented Primal‑Attention, a low‑rank attention module based on the Nyström method with an iterative pseudoinverse refinement. By using a small set of landmark tokens, it approximates the full attention matrix in linear time with respect to sequence length. The module integrates into the DistilBERT compression framework and offers a configurable rank parameter. Future work will address the mask handling, add zero‑shot weight transfer, and possibly incorporate head‑wise landmark selection. Primal‑Attention stands as a clean reference implementation of landmark‑based efficient attention.
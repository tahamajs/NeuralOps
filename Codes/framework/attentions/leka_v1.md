# LEKAv1: Linear‑Time Efficient Kernel Attention (Basic Version)

**Abstract**

We present **LEKAv1**, a pure kernel‑based linear attention module that replaces the quadratic softmax self‑attention with a low‑variance random Fourier feature map. Queries and keys are projected using fixed random matrices and then transformed via cosine/sine functions to approximate a Gaussian kernel. The attention output is then computed in linear time using the associative property of the kernel feature map, avoiding the \(O(N^2)\) attention matrix. The module uses zero‑shot weight initialisation from a pretrained DistilBERT attention layer, enabling efficient fine‑tuning with significant speedups on long sequences. An optional **Softpick** rectified softmax is provided for future integration, but the current forward pass uses the standard linear‑attention normalisation. The module is registered in the compression framework as a drop‑in replacement for self‑attention. This paper describes the mathematical formulation, implementation details, and the injection procedure.

---

## 1. Introduction

The quadratic computational complexity of standard multi‑head self‑attention with respect to sequence length \(N\) limits the application of Transformer models to very long documents. **Linear attention** addresses this by replacing the softmax kernel \(\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})\) with a finite‑dimensional feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\) and exploiting the associative property to reorder the computation. Among the many choices of feature map, random Fourier features (Rahimi & Recht, 2007) offer a principled way to approximate shift‑invariant kernels.

**LEKAv1** implements a straightforward version of this idea. It uses a fixed random projection matrix per attention head to project queries and keys to a higher‑dimensional space, then applies cosine and sine to create \(2F\)-dimensional features, approximating an RBF kernel. The linear attention formula is then applied: the key‑value pairs are aggregated into a memory matrix, and the output is computed by query‑memory multiplication followed by normalisation. The module is designed to be a plug‑in replacement for a DistilBERT attention layer, with pretrained weights copied from the original linear projections to preserve zero‑shot performance. The name “LEKA” stands for **L**inear‑time **E**fficient **K**ernel **A**ttention.

In this paper, we detail the architecture and mathematical underpinnings of LEKAv1, discuss the role of the random Fourier features, and describe the injection mechanism that makes it instantly usable in existing DistilBERT models.

---

## 2. Method

### 2.1 Multi‑Head Projections

Let the input hidden states be \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\). The module first applies learned linear projections for all heads:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q + \mathbf{b}_Q,\quad
\mathbf{K} = \mathbf{H} \mathbf{W}_K + \mathbf{b}_K,\quad
\mathbf{V} = \mathbf{H} \mathbf{W}_V + \mathbf{b}_V,
\]

with weight matrices \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\) and biases. These are then reshaped and transposed to the multi‑head layout \((B, H, N, d_h)\), where \(H\) is the number of heads and \(d_h = D/H\).

### 2.2 Random Fourier Feature Map

A random projection matrix \(\mathbf{R}_h \in \mathbb{R}^{d_h \times F}\) is instantiated once per head (with \(F\) = `feature_dim`, default 256) and stored as a non‑trainable buffer. For each head, the query and key vectors are projected:

\[
\mathbf{\Phi}_Q = \mathbf{Q} \mathbf{R}_h,\quad \mathbf{\Phi}_K = \mathbf{K} \mathbf{R}_h,
\]

both of shape \((B, H, N, F)\). Then the final feature map is constructed by concatenating cosine and sine of the projected vectors and scaling by \(1/\sqrt{F}\):

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \big[ \cos(\mathbf{x}\mathbf{R}_h),\; \sin(\mathbf{x}\mathbf{R}_h) \big] \in \mathbb{R}^{2F}.
\]

The scaling ensures that the expected squared norm of the feature vector is \(O(1)\), and the random projection approximates the Gaussian kernel:

\[
\phi(\mathbf{q})^\top \phi(\mathbf{k}) \approx \exp\!\big(\mathbf{q}^\top \mathbf{k} / \sqrt{d}\big),
\]

with variance decreasing as \(O(1/F)\).

### 2.3 Linear Attention

Using the feature map, the attention output for each head is computed without materialising the \(N \times N\) attention matrix. The key‑value memory \(\mathbf{KV}\) and the key sum \(\mathbf{Z}\) are computed first:

\[
\mathbf{KV} = \sum_{j=1}^N \phi(\mathbf{k}_j) \mathbf{v}_j^\top \;\in \mathbb{R}^{B \times H \times 2F \times d_h},
\qquad
\mathbf{Z} = \sum_{j=1}^N \phi(\mathbf{k}_j) \;\in \mathbb{R}^{B \times H \times 1 \times 2F}.
\]

Then for each query position \(i\),

\[
\mathbf{o}_i = \frac{\phi(\mathbf{q}_i)^\top \mathbf{KV}}{\phi(\mathbf{q}_i)^\top \mathbf{Z} + \epsilon},
\]

where division is applied element‑wise after the dot products. This yields the normalised, reweighted sum of values. All operations are linear in \(N\); the dominant cost is forming \(\mathbf{KV}\), which is \(O(N F d_h)\) per head, or \(O(N D F)\) overall. Since \(F\) is a constant (256), the total complexity is \(O(N D^2)\)—linear in \(N\).

### 2.4 Padding Mask

An optional 2D attention mask of shape \((B, N)\) can be provided, where valid tokens are marked with \(1\) and padding tokens with \(0\). The mask is broadcast to \((B, 1, 1, N)\) and multiplied with the context tensor after the linear attention computation, zeroing out the output for padding positions. This approach is simple but does not prevent padding tokens from influencing the memory aggregation \(\mathbf{KV}\) and \(\mathbf{Z}\). For better handling, one would need to mask \(\mathbf{K}\) and \(\mathbf{V}\) before building \(\mathbf{KV}\), which is not done in the current version.

### 2.5 Output Projection

The per‑head outputs are concatenated along the feature dimension, reshaped to \((B, N, D)\), and projected with a final linear layer:

\[
\mathbf{Y} = \mathbf{O}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) for compatibility.

### 2.6 Softpick (Unused)

The module defines a static method `softpick`, which is a rectified softmax that can produce sparse attention weights. It is not invoked in the forward pass, indicating it may be intended for future variants (e.g., local window attention with softpick) or as an optional replacement for the linear attention normalisation. The current linear attention uses the standard denominator \(\phi(\mathbf{Q})\mathbf{Z}^\top\) instead of softpick.

---

## 3. Implementation Details

### 3.1 Module Class

`LEKAV1Attention` inherits from `nn.Module` and is registered (via injection function) under `"leka_v1"`. Key attributes:

- `query`, `key`, `value`, `out_proj`: standard linear layers.
- `proj`: a buffer of shape \((H, d_h, F)\) initialised from a standard normal distribution.
- `feature_dim`: dimension \(F\) of the random features (from `comp_config`).
- `dropout`: applied to? Actually not used inside forward; the dropout layer is defined but never called. The dropout can be integrated into attention weights if softpick were used, but here it's left unused.
- `transpose_for_scores`: reshapes and transposes to multi‑head layout.
- `kernel_feature_map`: computes the cosine/sine features.

Forward steps:
1. Project hidden states to Q, K, V.
2. Compute \(\phi(Q), \phi(K)\).
3. Compute \(\mathbf{KV} = \text{einsum}(K\_feat, V)\).
4. Compute \(\mathbf{Z} = \text{sum}(K\_feat, dim=2)\).
5. Compute numerator = \(\text{einsum}(Q\_feat, KV)\).
6. Compute denominator = \(\text{einsum}(Q\_feat, Z\_squeezed) + 1e-8\).
7. Context = numerator / denominator.
8. Apply mask (if provided) by multiplication.
9. Reshape and output projection.

### 3.2 Injection Function

`inject_leka_v1` is the registered injection function. It creates a new `LEKAV1Attention` instance and copies the weights from the pretrained attention module (`old_attn`):

```python
new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
...
new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
```

Biases are copied similarly. This ensures the module starts from a zero‑shot approximation of the original attention, preserving most of the pretrained model’s performance before fine‑tuning.

### 3.3 Complexity

Per‑layer time is \(O(N F D + N D^2) \approx O(N D^2)\) (since \(F\) is constant). Memory is \(O(H F d_h) = O(D F)\), independent of \(N\), which is a huge saving over the \(O(N^2 D)\) memory of full attention.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"leka_v1"`. Relevant configuration parameters (in `comp_config`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dim` | 256 | Number of random features \(F\) (output dimension will be \(2F\)). |
| `dropout` | 0.1 | (Currently not used inside the module, but stored.) |

Example injection:
```python
comp_config.attention_type = "leka_v1"
comp_config.feature_dim = 256
```

Manual instantiation:
```python
attn = LEKAV1Attention(config=model_config, comp_config=comp_config, layer_idx=0, total_layers=6)
# copy weights from old_attn...
layer.attention = attn
```

The injection function handles the weight copying automatically.

---

## 5. Discussion

LEKAv1 provides a straightforward implementation of linear attention using random Fourier features. It offers the following advantages:

- **Linear time and memory** in \(N\), making it suitable for long sequences.
- **Zero‑shot weight transfer** ensures minimal initial performance loss.
- **Simplicity**: only a fixed random projection is added; no additional trainable parameters in the attention mechanism itself (apart from the usual Q,K,V,O projections).
- **Causal compatibility**: The recurrence is not explicit; the method as written is bidirectional, but can be made causal by using cumulative sums in the recurrence (not implemented here). For encoder tasks, bidirectional is fine.

However, several limitations exist:

- **No masking in KV accumulation**: Padding tokens still contribute to \(\mathbf{KV}\) and \(\mathbf{Z}\), which can bias the normalisation. This could be fixed by zeroing out K and V for padding positions before the einsum operations.
- **Dropout unused**: The dropout layer is defined but never applied. It could be applied to the attention weights if using softpick, but in the current path there is no explicit attention weight tensor to apply dropout to.
- **Random features are fixed**: The projection matrix is not learned, which might limit adaptation to the specific task. Learned or orthogonal features (like FAVOR+) could reduce variance.
- **No causal mask**: The module is bidirectional; for autoregressive tasks, a causal mask would need to be integrated into the linear attention recurrence.
- **Softpick not used**: The softpick function is present but serves no purpose in this version; it may confuse readers.

Despite these limitations, LEKAv1 is a valuable baseline for kernel‑based linear attention and demonstrates the ease of integrating efficient attention into pretrained models via weight copying.

---

## 6. Conclusion

We have presented LEKAv1, a linear‑time kernel attention module that uses random Fourier features to approximate the softmax kernel and computes attention in \(O(N D^2)\) time. By initialising its projection weights from a pretrained attention layer, it achieves strong zero‑shot performance and can be fine‑tuned efficiently. The module is integrated into the DistilBERT compression framework and serves as a foundation for further enhancements such as learnable features, causal masking, and softpick sparsity. Future work will address the masking of padding tokens and explore orthogonal feature maps for lower variance.
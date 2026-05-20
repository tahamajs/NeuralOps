# Griffin‑SSD Block: Hybrid Local Window Attention and Global State‑Space Duality Recurrence

**Abstract**

We introduce the **Griffin‑SSD Block**, a hybrid attention‑recurrent module that fuses a local causal sliding window self‑attention with a global causal linear attention based on the State‑Space Duality (SSD) kernel. The local branch captures fine‑grained, short‑range dependencies via exact scaled dot‑product attention within a fixed window, while the global branch employs a recurrent outer‑product state update that summarises the entire past history with linear time complexity. The outputs of both branches are summed and passed through a final projection. The design follows the Griffin architecture (De et al., 2024), which demonstrates that mixing local attention with a linear recurrent kernel can match the performance of full attention while being significantly faster for long sequences. This paper provides a detailed mathematical description of the local and global components, the implementation specifics, and the integration of the Griffin‑SSD Block as a drop‑in replacement for standard self‑attention in DistilBERT via the compression framework’s registry.

---

## 1. Introduction

The quadratic cost of multi‑head self‑attention limits its application to very long sequences. Recent work has shown that hybrid models, combining local attention windows with a lightweight global recurrent mechanism, can achieve an excellent trade‑off between efficiency and quality. The **Griffin‑SSD Block** adopts this hybrid paradigm. For each layer, the input sequence is processed through two parallel paths:

- A **local sliding window** attention that allows each token to attend to a limited number of previous tokens within a fixed window. This is a standard exact attention computation, providing high‑precision short‑range interactions.
- A **global State‑Space Duality (SSD) recurrence** that maintains a matrix state \(\mathbf{M}_t = \sum_{i=1}^t \mathbf{k}_i \mathbf{v}_i^\top\) and computes the output as \(\mathbf{q}_t \mathbf{M}_t\). This is a form of causal linear attention, equivalent to a state‑space model with a diagonal state transition, achieving linear complexity in sequence length.

The block is explicitly causal, making it suitable for autoregressive tasks; it processes tokens left‑to‑right. The local and global outputs are summed element‑wise, enabling the model to dynamically blend short‑term precision and long‑term memory.

In this paper, we detail the mathematical foundations of each component, the architecture of the Griffin‑SSD Block, and its integration into the DistilBERT compression framework. We also discuss computational complexity, limitations, and potential improvements.

---

## 2. Method

### 2.1 Notation and Projections

Let the input sequence be \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), where \(B\) is the batch size, \(L\) the sequence length, and \(D\) the hidden dimension. The block defines three projection matrices for the local branch and one additional projection for the global query, while sharing the key and value projections between both branches:

\[
\mathbf{Q}_{\text{local}} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]
\[
\mathbf{Q}_{\text{global}} = \mathbf{X} \mathbf{W}_{\text{ssd}},
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_{\text{ssd}} \in \mathbb{R}^{D \times D}\) (bias‑free). All tensors retain shape \((B, L, D)\).

### 2.2 Local Causal Window Attention

The local attention computes exact scaled dot‑product scores within a causal window. Raw scores are:

\[
\mathbf{S}^{\text{local}} = \frac{\mathbf{Q}_{\text{local}} \mathbf{K}^\top}{\sqrt{D}} \in \mathbb{R}^{B \times L \times L}.
\]

A binary mask is constructed that permits attention only to previous tokens whose distance is less than the window size \(W\):

\[
M_{ij} = \begin{cases}
1 & \text{if } j \le i \text{ and } i - j < W, \\
0 & \text{otherwise}.
\end{cases}
\]

Positions with \(M_{ij}=0\) are set to \(-\infty\) (implemented as \(-10^9\) in practice). The attention weights are then:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}^{\text{local}} + \log(-\infty) \cdot (1 - M)),
\]
\[
\mathbf{O}_{\text{local}} = \mathbf{A} \mathbf{V}.
\]

This operation is performed using batch matrix multiplication (`torch.bmm`), and since the window is limited to \(W\), the computational cost per token is \(O(W)\) rather than \(O(L)\). The causal constraint ensures no leakage from future tokens.

### 2.3 Global SSD Recurrence (Linear Attention)

The global branch computes a causal linear attention via the `ssd_kernel`. For each time step \(t\), we maintain a matrix state \(\mathbf{H}_t \in \mathbb{R}^{B \times D \times D}\) that accumulates the outer products of keys and values:

\[
\mathbf{H}_0 = \mathbf{0}, \qquad
\mathbf{H}_t = \mathbf{H}_{t-1} + \mathbf{k}_t \otimes \mathbf{v}_t,
\]

where \(\mathbf{k}_t = \mathbf{K}_{[:,t,:]} \in \mathbb{R}^{B \times D}\) and \(\mathbf{v}_t = \mathbf{V}_{[:,t,:]}\). The output at time \(t\) is then the product of the query \(\mathbf{q}_t = \mathbf{Q}_{\text{global}[:,t,:]}\) with the current state matrix:

\[
\mathbf{o}_t = \mathbf{q}_t \mathbf{H}_t \in \mathbb{R}^{B \times D}.
\]

Expanding the recurrence:

\[
\mathbf{o}_t = \sum_{i=1}^t \mathbf{q}_t (\mathbf{k}_i^\top \mathbf{v}_i) = \sum_{i=1}^t (\mathbf{q}_t \mathbf{k}_i^\top) \mathbf{v}_i.
\]

This is exactly the causal linear attention mechanism (Katharopoulos et al., 2020) without a denominator normalisation and with the identity feature map \(\phi(\mathbf{x}) = \mathbf{x}\). The state matrix is of size \(D \times D\) per batch element, and the per‑step update and query are \(O(D^2)\); the overall complexity is \(O(L D^2)\).

The `ssd_kernel` function implements this recurrence sequentially with a Python for‑loop. While not parallelised, it is a faithful implementation of the SSD kernel used in the Griffin architecture, which can be accelerated using parallel scan algorithms (Blelloch, 1990) to achieve \(O(\log L)\) time with suitable hardware.

### 2.4 Fusion and Output Projection

The final output of the block before projection is the element‑wise sum of the two branches:

\[
\mathbf{O} = \mathbf{O}_{\text{local}} + \mathbf{O}_{\text{global}}.
\]

An optional output projection \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times D}\) is then applied:

\[
\mathbf{Y} = \mathbf{O} \mathbf{W}_{\text{out}}.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) to conform to the framework’s attention interface, where the second element would normally be attention weights.

### 2.5 Relationship to Griffin and SSD

The Griffin architecture (De et al., 2024) uses a mix of local sliding window attention and a recurrent state‑space model (the SSD kernel) in a carefully scheduled pattern. Our block simplifies this by running both the local and global streams in parallel at every layer, summing their outputs. The SSD kernel is a direct simplification of the Mamba state‑space model: it uses no gating, no convolution, and no selection mechanism, just a plain linear accumulation of outer products. It corresponds to a linear attention with an identity kernel and no normalisation.

---

## 3. Implementation Details

The `GriffinSSDBlock` class extends `BaseAttention` (for compatibility with the compression framework) and is registered as `"griffin_ssd"`. Its constructor takes:

- `config`: provides `hidden_size`.
- `rank`: unused, kept for interface consistency.
- `window_size` (default 256): the size of the local causal window.

**Layers**:
- `W_q`, `W_k`, `W_v`: for local attention.
- `ssd_proj`: produces the global query.
- `out_proj`: final linear projection (with bias by default).

**Forward**:
1. Compute `q_local`, `k_local`, `v_local` using the respective projections.
2. Compute local attention scores with `torch.bmm`, apply causal window mask, softmax, and multiply with `v_local`.
3. Compute `q_global` with `ssd_proj`, then call `ssd_kernel(q_global, self.W_k(x), self.W_v(x))` to obtain the global output. Note that the same keys and values are used as for the local branch, but the function recalculates `W_k(x)` and `W_v(x)` (potentially recomputing the projections, which could be optimised).
4. Sum the two outputs and apply `out_proj`.
5. Return `(output, None)`.

The module does **not** apply dropout or use attention masking for padding. For padded sequences, the local window would still allow attention to padding tokens, and the global SSD recurrence would accumulate their keys/values, contaminating the state. This should be addressed in a production version by masking padding tokens during the recurrence update. The local attention also does not include an external `mask` parameter; the provided `mask` argument is ignored.

The sequential loop in `ssd_kernel` limits training speed for long sequences but is acceptable for moderate lengths; a future parallel scan implementation would be highly beneficial.

---

## 4. Usage and Configuration

The module can be injected into a DistilBERT layer via the registry key `"griffin_ssd"`. The relevant configuration parameters that can be passed through `comp_config` are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 256 | Width of the local causal window. |
| `rank` | (ignored) | Kept for interface compatibility. |

Example injection:
```python
comp_config.window_size = 256
# In the framework, attention type "griffin_ssd" will be instantiated.
```

Manual creation:
```python
attn = GriffinSSDBlock(config=model_config, window_size=512)
layer.attention = attn
```

---

## 5. Discussion

The Griffin‑SSD Block offers a compelling efficiency profile. The local window attention costs \(O(L W D)\) while the global SSD recurrence costs \(O(L D^2)\). For sequences where \(L \gg D\), the local attention dominates if \(W\) is small, and the overall complexity is roughly linear in \(L\). The SSD kernel provides a global receptive field without the quadratic blow‑up, and its additive combination with local attention ensures that both short‑range and long‑range information can be integrated.

However, the current implementation has several limitations. First, the lack of padding mask handling can cause information leakage from padding tokens; this can be fixed by zeroing out keys and values for padding positions during the recurrence update. Second, the sequential loop is slow on GPUs; replacing it with a parallel scan or an optimised linear attention kernel (e.g., `torch.nn.functional.scaled_dot_product_attention` for the local part and a batched matrix operation for the global part) would significantly improve performance. Third, the global linear attention uses an identity feature map and no normalisation, which may lead to unstable outputs for very long sequences; adding a denominator term or a learnable gating could stabilise training. Fourth, the block does not include multi‑head splitting; all operations are on the full hidden dimension, which may limit capacity. Multi‑head variants could be introduced by reshaping the projections.

Despite these issues, the module exemplifies the hybrid approach that is gaining traction in efficient Transformer design. Its simplicity and modularity make it a valuable baseline for further experimentation.

---

## 6. Conclusion

We have presented the Griffin‑SSD Block, a hybrid attention‑recurrent module that combines a local causal window attention with a global linear‑time SSD recurrence. The design provides a balanced trade‑off between local precision and global context, with linear overall complexity. Integrated into the DistilBERT compression framework, it serves as a ready‑to‑use efficient alternative to full self‑attention. Future work will address the masking and performance limitations, and explore multi‑head and normalised variants to approach the quality of full attention while maintaining high throughput.
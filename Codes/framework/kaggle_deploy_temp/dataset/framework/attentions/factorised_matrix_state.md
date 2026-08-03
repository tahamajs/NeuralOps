# Factorised Matrix State: A Linear Recurrent Attention Module

**Abstract**

We present **Factorised Matrix State (FMS)**, a lightweight recurrent module that emulates linearised self‑attention with \(O(N d^2)\) time complexity. The module maintains an evolving \(d \times d\) state matrix \(\mathbf{M}_t = \sum_{i=1}^{t} \mathbf{x}_i \mathbf{x}_i^\top\), where \(\mathbf{x}_i \in \mathbb{R}^d\) are the token representations. At each step, the output is computed as \(\mathbf{y}_t = \mathbf{M}_t \mathbf{x}_t\), which corresponds to a dot‑product attention over all past tokens without softmax. The state matrix is stored in full form, but the module also carries low‑rank factors \(\mathbf{L}, \mathbf{R}\) as a design hint for future compression. FMS can be used as a drop‑in replacement for self‑attention in Transformers, offering linear complexity in sequence length while retaining the ability to capture pairwise token interactions through a bilinear form. This paper describes the mathematical foundations, implementation details, and potential extensions of the FMS module.

---

## 1. Introduction

Transformers have become the backbone of modern sequence processing, but their quadratic self‑attention cost poses challenges for long sequences. Several approaches have attempted to reduce this cost, including kernelised linear attention, recurrence, and low‑rank approximations. The Factorised Matrix State module draws inspiration from linearised attention, specifically the view that self‑attention can be written as a recurrent update of a “memory” matrix. By storing the sum of outer products of all previous inputs, the model can compute a contextualised output for each new token in time proportional to \(d^2\) rather than \(t \cdot d\).

The module’s name hints at a potential compression of this state matrix using a low‑rank factorisation; while the current implementation maintains a full‑rank state, it includes learnable low‑rank factors \(\mathbf{L}, \mathbf{R} \in \mathbb{R}^{d \times r}\) that could later be used to approximate \(\mathbf{M}_t\). This paper details the core recurrent computation, its relationship to linear attention, and the role of the factorised parameters as a placeholder for further memory reduction.

---

## 2. Method

### 2.1 Recurrent State Update

Given an input sequence \(\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N)\) with \(\mathbf{x}_t \in \mathbb{R}^{d}\) (where \(d = d_{\text{model}}\) is the hidden dimension), the module maintains a symmetric matrix \(\mathbf{M}_t \in \mathbb{R}^{d \times d}\) that accumulates the outer products of the token embeddings seen so far:

\[
\mathbf{M}_0 = \mathbf{0},
\qquad
\mathbf{M}_t = \mathbf{M}_{t-1} + \mathbf{x}_t \mathbf{x}_t^\top, \quad t = 1, \dots, N.
\]

This recurrence can be seen as a “fast weight” memory that encodes pairwise interactions between tokens (Schmidhuber, 1992). The update is additive and does not require any forgetting mechanism; it simply grows with the sequence.

### 2.2 Output Computation

The output at time step \(t\) is defined as the matrix‑vector product of the current state and the current input:

\[
\mathbf{y}_t = \mathbf{M}_t \mathbf{x}_t.
\]

Expanding the definition of \(\mathbf{M}_t\),

\[
\mathbf{y}_t = \left( \sum_{i=1}^{t} \mathbf{x}_i \mathbf{x}_i^\top \right) \mathbf{x}_t = \sum_{i=1}^{t} \mathbf{x}_i (\mathbf{x}_i^\top \mathbf{x}_t).
\]

Thus, the output is a weighted sum of all previous token embeddings, where the weight of token \(i\) is the dot product \(\mathbf{x}_i^\top \mathbf{x}_t\). This is exactly the standard linear attention mechanism (without scaling and softmax) when causality is enforced by the recurrent state. Indeed, linear transformers (Katharopoulos et al., 2020) compute:

\[
\mathbf{y}_t = \frac{\sum_{i=1}^t \phi(\mathbf{x}_i) (\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_t))}{\sum_{i=1}^t \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_t)},
\]

using a kernel feature map \(\phi\). The FMS module is a special case where \(\phi(\mathbf{x}) = \mathbf{x}\) (the identity map) and no denominator normalisation is applied. While this removes the non‑linearity and probabilistic interpretation of softmax, it retains the core ability to mix information based on dot‑product similarity.

### 2.3 Low‑Rank Placeholder

The module also declares two parameter matrices \(\mathbf{L}, \mathbf{R} \in \mathbb{R}^{d \times r}\) with a default rank \(r = 32\). In the current implementation, these parameters are initialised but **not used** in the forward pass. Their presence indicates the intention to eventually compress the \(d \times d\) state matrix into a low‑rank form, e.g.,

\[
\mathbf{M}_t \approx \mathbf{L} \mathbf{S}_t \mathbf{R}^\top,
\]

where \(\mathbf{S}_t \in \mathbb{R}^{r \times r}\) would be updated recurrently. This would reduce the memory and computational cost from \(O(d^2)\) to \(O(dr + r^2)\). At present, however, the full state is maintained.

---

## 3. Implementation Details

The `FactorisedMatrixState` class is registered as `"factorised_matrix_state"` in the attention registry. Key implementation aspects:

- **Constructor**:
  - `config`: provides `hidden_size` (\(d\)).
  - `rank`: the rank of the factorised parameters (default 32). Used to initialise `self.L` and `self.R` as \(d \times r\) matrices with small random values (std 0.01).
- **Forward pass**:
  1. Accepts `hidden_states` of shape \((B, N, d)\).
  2. Initialises a state tensor `state` of shape \((B, d, d)\) to zeros.
  3. Iterates over the time dimension \(t = 0, \dots, N-1\):
     - Extracts \(\mathbf{x}_t\) of shape \((B, d)\).
     - Updates the state: `state = state + outer(xt, xt)`, i.e., \(\mathbf{M}_t = \mathbf{M}_{t-1} + \mathbf{x}_t \mathbf{x}_t^\top\).
     - Computes output: `out_t = state @ xt.unsqueeze(-1)` and removes the extra dimension, giving \((B, d)\).
  4. Stacks the per‑step outputs along the time axis to produce a tensor of shape \((B, N, d)\).
  5. Returns a tuple `(out, None)` to mimic the standard attention output format.

- **Mask handling**: The `attention_mask` argument is accepted but not used; masking would need to be incorporated to ignore padding tokens (e.g., by not updating the state for padded positions). The current version assumes no padding.

- **Training**: All operations are differentiable; gradients flow through the state updates and the linear projections.

---

## 4. Usage and Configuration

The module can be inserted into a Transformer layer as an alternative to standard self‑attention. The registry key is `"factorised_matrix_state"`. Configuration is minimal:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 32 | Rank of the placeholder low‑rank factors (unused in forward). |

Example instantiation:

```python
fms = FactorisedMatrixState(config=model_config, rank=32)
# Replace layer.attention with fms
```

The module expects the input to be the hidden states of shape \((B, N, d)\). No additional parameters are needed. For production use, one should add support for padding masks and possibly a normalisation denominator.

---

## 5. Discussion

The Factorised Matrix State module offers several attractive properties:

- **Linear complexity**: \(O(N d^2)\) time and \(O(d^2)\) memory (independent of \(N\)), making it suitable for long sequences.
- **Causality**: By construction, the output at step \(t\) depends only on past tokens, making it directly applicable to autoregressive generation without a causal mask.
- **Simplicity**: The recurrence is straightforward, requiring no attention scores, softmax, or kernel approximations. It can be seen as a learnable, content‑dependent linear transform of the sequence.

However, there are notable limitations:

- **Lack of normalisation**: The absence of a denominator (as in linear attention) can lead to exploding outputs for long sequences because the state matrix entries grow with the sum of squared embeddings. A normalising term (e.g., dividing by the sum of dot products) would stabilise training but is not included.
- **Full state storage**: The \(d \times d\) state matrix, while not dependent on \(N\), can still be large for high‑dimensional models (e.g., \(d=1024\) yields a million‑entry matrix). The intended low‑rank compression using \(\mathbf{L}, \mathbf{R}\) would address this but is not yet implemented.
- **No non‑linearity**: The identity feature map may be insufficient to capture complex interactions; the original linear attention introduced a non‑linear feature map \(\phi(x) = \text{elu}(x)+1\) to approximate softmax. The FMS could be extended with such a feature map.
- **Padding**: The current sequential loop processes all tokens equally, including padding. A proper masking mechanism would require skipping state updates for padded positions, which is straightforward to add.

Despite these drawbacks, FMS serves as a pedagogical example of the connection between linear attention and recurrent networks. With the addition of normalisation and a kernel feature map, it could become a competitive efficient attention module.

---

## 6. Conclusion

We have introduced Factorised Matrix State, a recurrent module that computes a linearised form of self‑attention by accumulating outer products of token embeddings. The module is simple, causal, and of linear complexity, making it a promising building block for efficient Transformers. Future work will implement the hinted low‑rank state compression and integrate normalisation and feature maps to approach the performance of standard attention while maintaining scalability. The module’s plug‑and‑play design allows immediate experimentation within existing Transformer frameworks.
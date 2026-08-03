# HGRN2‑Outer: A Gated Matrix‑State Linear Recurrence

**Abstract**

We introduce the **HGRN2‑Outer Block**, a recurrent sequence mixing module that replaces standard self‑attention with a gated, outer‑product‑based memory. At each time step, the module updates a square matrix state \(\mathbf{H}_t \in \mathbb{R}^{D \times D}\) using a vector‑wise forget gate \(\mathbf{f}_t\) and the outer product of learnable key and value projections. The output for a token is the product of the state matrix with the token’s query (input itself), i.e., \(\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t\). This design is a direct extension of the HGRN2 recurrence from a vector state to a matrix state, enabling richer pairwise token interactions to be stored in the memory. The forget gate applies per‑row scaling, allowing each dimension of the memory to decay at a different rate. The module has linear time complexity in sequence length \(O(L D^2)\) and serves as a drop‑in replacement for the self‑attention layer in Transformer models. It is integrated into the DistilBERT compression framework via a registry‑based injection function.

---

## 1. Introduction

Self‑attention in Transformers captures token‑to‑token interactions via a full \(L \times L\) attention matrix, leading to quadratic complexity. Linearised attention methods replace the softmax with a kernel feature map, enabling the output to be computed using a recurrent state of shape \(D \times D\) (for the key‑value outer product) at linear cost. The **HGRN2‑Outer Block** builds on this idea, but introduces a **gated update** inspired by the HGRN2 (Hierarchically Gated Recurrent Network) philosophy. Unlike standard linear attention which accumulates key‑value pairs without any forgetting, HGRN2‑Outer uses a learnable **forget gate** to control the retention of each row of the memory matrix independently. This provides a flexible mechanism to discard outdated information and better adapt to long sequences.

The module maintains a matrix state \(\mathbf{H}_t \in \mathbb{R}^{D \times D}\) that is updated recursively. At step \(t\), the input \(\mathbf{x}_t\) is projected to key \(\mathbf{k}_t\) and value \(\mathbf{v}_t\), and the forget gate \(\mathbf{f}_t \in \mathbb{R}^D\) is computed from \(\mathbf{x}_t\). The state update combines the previous state, scaled row‑wise by \(\mathbf{f}_t\), with the outer product \(\mathbf{k}_t \mathbf{v}_t^\top\) scaled by \(1-\mathbf{f}_t\). The output is then \(\mathbf{H}_t \mathbf{x}_t\), which implicitly computes a weighted sum of past values. The entire operation is recurrent and causal, making it suitable for autoregressive tasks.

In this paper, we detail the mathematical formulation of the HGRN2‑Outer recurrence, its implementation as a PyTorch module, and its integration into the compression framework. We also discuss its relationship to other linear attention variants and its potential advantages.

---

## 2. Method

### 2.1 Notation

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be a batch of sequences with length \(L\) and hidden dimension \(D\). For each time step \(t\), let \(\mathbf{x}_t = \mathbf{X}_{[:,t,:]} \in \mathbb{R}^{B \times D}\). The module maintains a **matrix state** \(\mathbf{H}_t \in \mathbb{R}^{B \times D \times D}\), initialised to the zero matrix: \(\mathbf{H}_0 = \mathbf{0}\).

### 2.2 Projections and Forget Gate

At each step, the current token \(\mathbf{x}_t\) is linearly projected to a key and a value:

\[
\mathbf{k}_t = \mathbf{W}_k \mathbf{x}_t + \mathbf{b}_k, \qquad
\mathbf{v}_t = \mathbf{W}_v \mathbf{x}_t + \mathbf{b}_v,
\]

where \(\mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{D \times D}\) and biases \(\mathbf{b}_k, \mathbf{b}_v\) are learnable parameters (the code uses `nn.Linear` with default bias). Additionally, a forget gate vector \(\mathbf{f}_t \in \mathbb{R}^{B \times D}\) is computed from the input using a separate linear layer followed by a sigmoid activation:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f).
\]

### 2.3 Gated Matrix State Update

The state is updated as a **row‑wise interpolation** between the previous state and the new outer product. Specifically, for each row index \(i \in \{1,\dots,D\}\),

\[
\mathbf{H}_{t,i} = \mathbf{f}_{t,i} \cdot \mathbf{H}_{t-1,i} + (1 - \mathbf{f}_{t,i}) \cdot (\mathbf{k}_t \mathbf{v}_t^\top)_i,
\]

where \(\mathbf{H}_{t,i}\) denotes the \(i\)-th row of the matrix state, and \(\mathbf{f}_{t,i}\) is the \(i\)-th component of the forget gate. In tensor form, this is written as:

\[
\mathbf{H}_t = \mathbf{f}_t \!\uparrow\! \odot \mathbf{H}_{t-1} + (1 - \mathbf{f}_t\!\uparrow\!) \odot (\mathbf{k}_t \!\downarrow\! \mathbf{v}_t^\top),
\]

where \(\mathbf{f}_t\!\uparrow\!\) adds a trailing dimension (size 1) to broadcast along columns, and \(\mathbf{k}_t\!\downarrow\!\) adds a trailing dimension for outer product. Concretely, in the code:

- \(\mathbf{f}_t\).unsqueeze(-1) → shape \((B, D, 1)\).
- \(\mathbf{k}_t\).unsqueeze(-1) → shape \((B, D, 1)\); \(\mathbf{v}_t\).unsqueeze(1) → shape \((B, 1, D)\); their matrix product gives \((B, D, D)\).

Thus, the update formula used in the forward pass is:

\[
\mathbf{H}_t = \mathbf{f}_t\texttt{.unsqueeze(-1)} * \mathbf{H}_{t-1} \;+\; (1 - \mathbf{f}_t\texttt{.unsqueeze(-1)}) * \big( \mathbf{k}_t\texttt{.unsqueeze(-1)} \mathbf{v}_t\texttt{.unsqueeze(1)} \big).
\]

This operation scales each row of the previous state by the corresponding forget component, and adds a fraction of the new outer product.

### 2.4 Output Readout

After updating the state, the output for time \(t\) is produced by multiplying the state matrix with the current token (acting as a query) and reducing the last dimension:

\[
\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t^{\!\uparrow} \quad \text{(squeezed)}.
\]

In code, this is `(state @ xt.unsqueeze(-1)).squeeze(-1)`, giving shape \((B, D)\). Expanding the recurrence reveals that \(\mathbf{y}_t = \sum_{i=1}^t \alpha_{t,i} (\mathbf{x}_t^\top \mathbf{k}_i) \mathbf{v}_i\), where the weights \(\alpha_{t,i}\) depend on the product of forget gates from \(i+1\) to \(t\). Thus, it implements a type of gated linear attention with a soft forgetting mechanism.

The outputs for all time steps are stacked into \(\mathbf{Y} \in \mathbb{R}^{B \times L \times D}\) and passed through a final output projection:

\[
\mathbf{O} = \mathbf{W}_o \mathbf{Y} + \mathbf{b}_o,
\]

with \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\). The module returns a tuple \((\mathbf{O}, \text{None})\) for compatibility with the framework’s attention interface.

### 2.5 Complexity

Each step involves:
- Three linear transformations of size \(D \times D\): cost \(O(D^2)\).
- An outer product of two vectors of size \(D\): cost \(O(D^2)\).
- A matrix‑vector multiplication for the output: cost \(O(D^2)\).

Thus, the total time for a sequence of length \(L\) is \(O(L D^2)\), which is linear in \(L\). The memory required for the state is \(O(D^2)\) per batch element, independent of sequence length.

---

## 3. Implementation

### 3.1 Module Definition

The `HGRN2OuterBlock` class is defined as a `nn.Module`. It stores:
- `self.W_f`: linear layer for forget gate.
- `self.W_k`, `self.W_v`: linear layers for key and value.
- `self.out_proj`: final output projection.

All layers include biases by default.

### 3.2 Forward Pass

The `forward` method accepts `x` (shape \(B \times L \times D\)) and an optional `state` (to allow continuation of sequences across calls). If no state is provided, it initialises a zero tensor of shape \((B, D, D)\). It then loops over time steps, updates the state using the gated formula, computes the output token, and stacks them. Finally, it applies the output projection and returns `(output, None)`.

The method also accepts `**kwargs` (e.g., `attention_mask`) but ignores them; padding tokens are not masked, meaning they will affect the state. This should be addressed in a production‑ready implementation.

### 3.3 Injection Function

The function `inject_hgrn2_outer` is registered under the key `"hgrn2_outer"` in the `ATTENTION_REGISTRY`. It receives the layer, model config, compression config, and the original attention module. It extracts `hidden_size` from the model config, creates an `HGRN2OuterBlock(dim)`, and assigns it to `layer.attention`. No pretrained weight transfer is performed; the module is trained from scratch.

---

## 4. Usage and Configuration

The block can be injected by setting the attention type to `"hgrn2_outer"` in the compression configuration. No additional hyperparameters are required; the hidden size is taken from the model config.

Example injection:
```python
comp_config.attention_type = "hgrn2_outer"
```

Manual instantiation:
```python
block = HGRN2OuterBlock(dim=768)
layer.attention = block
```

The outer residual connection and LayerNorm in the Transformer layer still apply, so the full transformation is:

\[
\mathbf{H}_{\text{out}} = \text{LayerNorm}(\mathbf{H} + \text{HGRN2OuterBlock}(\mathbf{H})).
\]

---

## 5. Discussion

The HGRN2‑Outer Block generalises the vector‑state HGRN2 to a matrix state, allowing the model to store pairwise feature interactions rather than just weighted sums of token vectors. The per‑row forget gate provides a fine‑grained forgetting mechanism that can learn to retain certain dimensions of the memory over long periods while rapidly overwriting others. This is akin to a state‑space model with diagonal state transition, but with an outer‑product update that captures second‑order statistics.

Compared to standard linear attention (which accumulates \(\mathbf{k}_t \mathbf{v}_t^\top\) without any forgetting), the gated mechanism prevents the memory from saturating and gives older information less influence over time. Compared to the HGRN2 vector recurrence, the matrix state can model richer token relationships because it stores a full covariance matrix instead of just a running sum.

However, several limitations exist. The sequential Python loop over time steps impedes training parallelisation; a parallel scan or a vectorised associative scan implementation would be needed for high speed. The \(D \times D\) state can be large for high‑dimensional models (e.g., \(D=1024\) gives over a million elements per batch element). Low‑rank approximations could reduce this cost. The module also ignores padding masks, making it unsuitable for batched sequences with variable lengths without explicit masking. Finally, the state is initialised to zero and the forget gate is applied from the very first token; the initial bias of the forget gate (default linear bias zero → initial \(\mathbf{f} \approx 0.5\)) may need tuning for optimal training.

Despite these issues, the HGRN2‑Outer Block offers a compelling blend of simplicity, linear complexity, and powerful state representation. It is a valuable baseline for exploring gated matrix recurrences as efficient alternatives to attention.

---

## 6. Conclusion

We have presented the HGRN2‑Outer Block, a gated matrix‑state recurrent module for sequence mixing. By combining a per‑row forget gate with an outer‑product update, it captures second‑order token interactions while maintaining linear time complexity. The module integrates seamlessly into the DistilBERT compression framework and serves as a drop‑in replacement for self‑attention. Future work will focus on parallelisation, masking, and low‑rank approximations to enhance both speed and capacity. The HGRN2‑Outer Block provides a clear example of how gating can be applied to matrix memories to build efficient, powerful sequence models.
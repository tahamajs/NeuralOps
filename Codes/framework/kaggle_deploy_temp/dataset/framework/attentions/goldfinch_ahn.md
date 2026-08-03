# GoldFinch‑AHN Block: Short‑Term Sliding Window + Long‑Term Gated DeltaNet Memory

**Abstract**

We introduce the **GoldFinch‑AHN Block**, a hybrid attention‑recurrent module that combines a short‑term causal sliding window attention with a long‑term memory based on a Gated DeltaNet layer. A learned scalar gate dynamically interpolates between the two information sources for each token, allowing the model to flexibly adjust its reliance on recent local context versus a persistent global memory. The block is designed as a drop‑in replacement for standard self‑attention in Transformers, and it aims to capture both fine‑grained local dependencies and long‑range temporal structure while maintaining linear complexity in the window size and memory size. The short‑term attention is computed using exact scaled dot‑product within a fixed window, while the long‑term stream uses a recurrent network (placeholder for GatedDeltaNet) that processes the sequence sequentially or via a fast update rule, keeping a compressed state. This paper details the architecture, the mathematical formulation of the windowed attention and the gating mechanism, and the integration into a DistilBERT‑style framework.

---

## 1. Introduction

Self‑attention mechanisms in Transformers allow each token to attend to all others, but the quadratic cost has motivated hybrid designs that combine a local attention window with a global memory. The GoldFinch‑AHN (Adaptive Hybrid Network) Block implements such a hybrid by splitting the information flow into two parallel streams:

- A **short‑term sliding window** attention that uses a causal mask and a local window to capture immediate context.
- A **long‑term memory** realised through a Gated DeltaNet layer (or a placeholder thereof), which maintains a recurrent state that summarizes all past tokens.

A token‑wise gate blends the outputs of the two streams, giving the model the capacity to decide, at each position, whether to rely on recent history or on the long‑term memory. This design is motivated by the observation that many language tasks require both local syntax (well captured by a narrow window) and topic‑level coherence (requiring a longer memory), and a simple gate can successfully combine them.

In the following sections, we detail the method, present the mathematical equations, discuss the placeholder nature of the GatedDeltaNetLayer, and describe the implementation.

---

## 2. Method

### 2.1 Projections

Given an input sequence \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\) (batch size \(B\), sequence length \(L\), hidden size \(D\)), the block first computes three linear projections without bias:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{H} \mathbf{W}_V,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{V}_Q \in \mathbb{R}^{D \times D}\). Note that the projections are applied to the full hidden dimension without splitting into multiple heads; the block operates in a “single‑head” fashion (or equivalently, all heads are fused). This simplifies the design and is suitable when the overall model already has multiple layers.

### 2.2 Short‑Term Causal Sliding Window Attention

The short‑term output is computed using a causal windowed attention. The raw attention scores are:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{D}} \quad \in \mathbb{R}^{B \times L \times L}.
\]

A binary mask is constructed that allows attention only to past tokens within a window of size \(W\) (the `window_size` hyperparameter). Formally, for position \(i\), the allowed keys are those positions \(j\) such that:

\[
0 \le i - j \le W-1 \quad \text{and} \quad j \le i.
\]

The mask is defined as:

\[
M_{ij} = \begin{cases}
1 & \text{if } j \le i \text{ and } i - j < W,\\
0 & \text{otherwise}.
\end{cases}
\]

Positions with \(M_{ij} = 0\) have their scores set to \(-\infty\). The attention weights are then:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \log(-\infty) \cdot (1 - M)),
\]

and the short‑term output is:

\[
\mathbf{O}_{\text{short}} = \mathbf{A} \mathbf{V}.
\]

This is analogous to a local, causal window commonly used in autoregressive models; it restricts each token’s view to its immediate left context, reducing the complexity from \(O(L^2 D)\) to \(O(L W D)\).

### 2.3 Long‑Term Memory via GatedDeltaNetLayer

The long‑term stream processes the input sequence through a recurrent layer called `GatedDeltaNetLayer`. In the provided code, this layer is a placeholder—simply a linear projection—but the intended design is a gated recurrent unit with a state that carries information from the past. In a full implementation, the GatedDeltaNet would maintain a hidden state \(\mathbf{c}_t \in \mathbb{R}^D\) that evolves as:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{c}_{t-1}),
\]
\[
\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{c}_{t-1}),
\]
\[
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{c}_{t-1}),
\]
\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
\]
\[
\mathbf{h}_t = \mathbf{c}_t.
\]

However, here the placeholder is a simple linear layer, so the long‑term output is just \(\mathbf{O}_{\text{long}} = \mathbf{x} \mathbf{W}_{\text{delta}}\). For the purpose of explanation, we treat it as a black‑box recurrent memory that produces a tensor of shape \((B, L, D)\) summarising the past.

### 2.4 Adaptive Gating

A token‑wise gate \(\mathbf{g} \in \mathbb{R}^{B \times L \times 1}\) is computed from the original hidden states:

\[
\mathbf{g} = \sigma(\mathbf{H} \mathbf{w}_g + b_g),
\]

where \(\mathbf{w}_g \in \mathbb{R}^{D \times 1}\) and \(b_g\) is a bias. The gate is then used to interpolate between the short‑term and long‑term outputs:

\[
\mathbf{O} = \mathbf{O}_{\text{short}} \odot \mathbf{g} + \mathbf{O}_{\text{long}} \odot (1 - \mathbf{g}).
\]

This convex combination allows each token to dynamically balance its reliance on immediate context versus the broader memory.

### 2.5 Output Projection

The blended output is projected back to dimension \(D\) using an output matrix \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\):

\[
\mathbf{Y} = \mathbf{O} \mathbf{W}_O.
\]

The final output is returned as a tuple \((\mathbf{Y}, \text{None})\) for compatibility with the framework’s attention interface.

---

## 3. Implementation Details

### 3.1 GoldFinchAHNBlock Class

- **Inheritance**: Inherits from `BaseAttention`, which may provide additional hooks (e.g., orthogonal loss).
- **Constructor parameters**:
  - `config`: contains `hidden_size` (\(D\)).
  - `rank`: unused, kept for interface consistency.
  - `window_size` (default 256): the size of the sliding window.
- **Layers**:
  - `W_q`, `W_k`, `W_v`: linear projections for query, key, value.
  - `delta_mem`: instance of `GatedDeltaNetLayer` (placeholder). The constructor passes `use_spectral_norm=True`, suggesting that the full implementation may use spectral normalisation for stability.
  - `gate`: linear layer \(D \rightarrow 1\).
  - `out_proj`: final linear projection.

- **Forward**:
  1. Computes Q, K, V.
  2. Computes attention scores using batch matrix multiplication: `torch.bmm(q, k.transpose(1, 2)) / sqrt(D)`. The tensors are of shape \((B, L, D)\).
  3. Constructs a causal window mask using `torch.arange(L)`. The mask `causal` is lower‑triangular; `window` restricts to indices where \(i - j < W\). The combined mask is their logical AND. Scores are masked with `-inf` for invalid positions.
  4. Applies softmax to get attention weights and computes `short_out = bmm(attn, v)`.
  5. Computes `long_out = self.delta_mem(x)`.
  6. Computes gate `g = sigmoid(gate(x))`.
  7. Blends: `out = out_proj(g * short_out + (1-g) * long_out)`.
  8. Returns `(out, None)`.

The module does not include dropout in the provided snippet; it could be added after the gate or in the attention.

### 3.2 Placeholder GatedDeltaNetLayer

The stub `GatedDeltaNetLayer` is a single linear layer (`net = nn.Linear(d_model, d_model)`). Its `forward` just applies the linear transform, meaning the “long‑term memory” in this implementation is a simple linear projection of the input. While this serves as a placeholder, the design intention is to replace it with a true recurrent or linear‑attention‑based memory that accumulates information over time. The flag `use_spectral_norm` is accepted but not used in the stub.

### 3.3 Integration

The block is registered under `"goldfinch_ahn"` in `ATTENTION_REGISTRY`. The injection function (not shown) would replace the standard attention layer with this block, likely initialising the projections from pretrained weights if needed.

---

## 4. Usage and Configuration

The block can be injected via the framework’s configuration by setting the attention type to `"goldfinch_ahn"`. The relevant hyperparameters that can be passed through `comp_config` are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 256 | Width of the sliding window for short‑term attention. |
| `rank` | (ignored) | Kept for interface compatibility. |

No other special configuration is needed for the placeholder version. To upgrade to a true long‑term memory, one would replace the stub with a proper GatedDeltaNet implementation.

---

## 5. Discussion

The GoldFinch‑AHN block offers a lightweight hybrid model that is particularly suited for long sequences. The short‑term window ensures that local dependencies are captured with high precision, while the long‑term stream provides a global context that does not grow in computational cost with distance. The gate, conditioned on the token’s hidden state, allows the model to learn when to trust the recent window (e.g., for syntactic agreement) versus when to rely on the memory (e.g., for topic tracking or coreference).

A key limitation is that the current long‑term memory is just a linear projection, which does not actually maintain a stateful memory. The full GatedDeltaNetLayer should be incorporated to realise the intended benefits. Additionally, the block operates on a single‑head, full‑dimensional representation, which may limit its capacity compared to multi‑head attention. Splitting the projection into multiple heads could improve expressiveness. The causal mask is hard‑coded, making the block suitable only for autoregressive tasks; for bidirectional encoding, the causal condition would need to be removed. Finally, padding masks are not handled in the current code; tokens beyond the sequence length would be attended to in the short‑term attention unless explicitly masked.

Despite these gaps, the block provides a clear template for exploring hybrid attention‑recurrent designs, and with a proper long‑term memory, it could match or exceed the performance of full attention on long‑range tasks while remaining efficient.

---

## 6. Conclusion

We have described the GoldFinch‑AHN Block, a hybrid module that marries a causal sliding window attention with a gated recurrent memory. The token‑wise gating mechanism dynamically fuses the two sources, offering a balance between local precision and global context. The placeholder GatedDeltaNetLayer marks the spot for a future recurrent memory, and when fully implemented, the block promises linear complexity in both sequence length and memory size. Its integration into the DistilBERT framework via the registry makes it a convenient experimental tool for efficient sequence modelling.
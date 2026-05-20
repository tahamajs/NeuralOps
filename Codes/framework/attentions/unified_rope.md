# UnifiedRoPE: A Universal Rotary Position Embedding for Consistent Positional Encoding in Hybrid Architectures

**Abstract**

We present **UnifiedRoPE**, a standalone module that implements **Rotary Position Embedding (RoPE)** as a universal positional encoding layer. Its primary purpose is to provide a single, consistent source of rotational position information that can be shared across multiple branches of a hybrid architecture—such as attention and state‑space model (SSM) pathways—ensuring that all sequence mixing components perceive the same relative position geometry. The module precomputes sinusoidal rotation matrices for all positions up to a maximum sequence length and applies the characteristic pairwise rotation: for each position, the even and odd feature dimensions are rotated by an angle proportional to their frequency band and the token’s position. The rotation formula is \(\operatorname{RoPE}(\mathbf{x}, t) = \mathbf{x} \odot \cos(\boldsymbol{\theta} t) + \operatorname{rotate\_half}(\mathbf{x}) \odot \sin(\boldsymbol{\theta} t)\), where \(\boldsymbol{\theta}_i = 10000^{-2i/d_{\text{model}}}\). UnifiedRoPE can be inserted into any Transformer or hybrid block to inject rotary position information without modifying the underlying sequence mixing logic. The module is registered in the compression framework under the key `"compression_unified_rope"`. This paper provides a rigorous mathematical derivation, implementation details, and a discussion of its role in achieving positional consistency across disparate branches.

---

## 1. Introduction

Position information is crucial for Transformers to process sequence order. Rotary Position Embedding (RoPE) (Su et al., 2024) has become a widely adopted method because it encodes relative position directly into the dot‑product of queries and keys, improving extrapolation to longer sequences. In hybrid architectures that combine attention with other sequence mixing mechanisms (such as state‑space models or convolutions), a common challenge is to provide a unified positional representation: the attention branch typically receives RoPE‑transformed queries and keys, but the non‑attention branch may operate on raw features without position information. This inconsistency can lead to sub‑optimal integration.

The **UnifiedRoPE** module addresses this by implementing RoPE as a self‑contained, pre‑computed rotation that can be applied to the full hidden states **before** they are fed into any sub‑module. By applying the same rotation to all branches, the model ensures that every pathway—attention, SSM, gating networks, etc.—shares an identical positional encoding. This “unified” approach simplifies the design of hybrid blocks and encourages positional coherence throughout the model.

In this paper, we mathematically define the RoPE transformation, detail the precomputation of rotation frequencies, describe the module’s implementation, and discuss its role in the compression framework. The module is designed to be easily integrated into any Transformer layer by simply wrapping the input with the unified rotation.

---

## 2. Method

### 2.1 Rotary Position Embedding (RoPE)

The RoPE technique encodes position \(t\) by applying a rotation matrix \(\mathbf{R}_t \in \mathbb{R}^{d \times d}\) to the feature vector \(\mathbf{x} \in \mathbb{R}^d\), where \(d\) is the hidden dimension. The rotation is structured to operate on consecutive pairs of dimensions. For a feature vector of even dimension, the rotation is defined as:

\[
\operatorname{RoPE}(\mathbf{x}, t) = \mathbf{x} \odot \mathbf{C}_t + \operatorname{rotate\_half}(\mathbf{x}) \odot \mathbf{S}_t,
\]

where

\[
\mathbf{C}_t = \big[\cos(\theta_0 t), \cos(\theta_0 t), \cos(\theta_1 t), \cos(\theta_1 t), \dots \big],
\]
\[
\mathbf{S}_t = \big[\sin(\theta_0 t), \sin(\theta_0 t), \sin(\theta_1 t), \sin(\theta_1 t), \dots \big],
\]

and the frequencies are given by

\[
\theta_i = \text{base}^{-2i / d}, \quad i = 0, 1, \dots, \frac{d}{2} - 1,
\]

with a default base of \(10000\). The function \(\operatorname{rotate\_half}\) swaps the first half of the vector with the negated second half:

\[
\operatorname{rotate\_half}([x_1, x_2, \dots, x_{d/2}, x_{d/2+1}, \dots, x_d]) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}].
\]

This definition ensures that for any positions \(m, n\), the inner product satisfies:

\[
\operatorname{RoPE}(\mathbf{q}, m)^\top \operatorname{RoPE}(\mathbf{k}, n) = \mathbf{q}^\top \mathbf{k} \cos(\theta(m-n)) + \text{rotation interference term},
\]

which effectively embeds the relative position \(m-n\) into the dot product.

### 2.2 Unified Application

In a hybrid block, the input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times d}\) are first passed through the UnifiedRoPE module:

\[
\tilde{\mathbf{X}} = \operatorname{RoPE}(\mathbf{X}).
\]

The rotated states are then distributed to various branches: e.g., for queries and keys in attention, and as input to a state‑space model or gating network. Because the rotation is applied once and shared, all branches operate on position‑aware representations.

### 2.3 Frequency Precomputation

To avoid recomputing trigonometric values at each forward pass, the module precomputes the \(\cos\) and \(\sin\) tables for a maximum sequence length \(L_{\max}\) (default 4096). The frequency vector \(\boldsymbol{\omega} \in \mathbb{R}^{d/2}\) is stored as a buffer. For a given sequence length \(L\), the rotation angles for all positions form a matrix \(\boldsymbol{\Theta} \in \mathbb{R}^{L \times d/2}\) where \(\Theta_{t,i} = t \cdot \omega_i\). The cosine and sine are then cached as tensors of shape \((L, d)\). During the forward pass, the appropriate slice is taken and broadcast to the batch size and head dimensions (if needed). The caching mechanism automatically rebuilds the table if the input sequence exceeds the current maximum length, ensuring dynamic adaptability.

### 2.4 Mathematical Stability and Implementation

The rotation is applied element‑wise using broadcasting. The \(\cos\) and \(\sin\) tensors are reshaped to \((1, L, 1, d)\) to align with input shape \((B, L, d)\). The rotation formula becomes:

\[
\mathbf{X}_{\text{rot}} = \mathbf{X} \odot \cos(\boldsymbol{\Theta}) + \text{rotate\_half}(\mathbf{X}) \odot \sin(\boldsymbol{\Theta}).
\]

The static method `_rotate_half` performs the swap and negation. The entire operation is differentiable and compatible with mixed‑precision training.

---

## 3. Implementation Details

- **Class `UnifiedRoPE`** inherits from `nn.Module` and is registered under `"compression_unified_rope"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `dim`: feature dimension \(d\) (must be even).
  - `max_seq_len`: maximum sequence length for precomputation (default 4096).
  - `base`: base frequency for the inverse frequency computation (default 10000).
  - Computes `inv_freq` as a 1D tensor of length \(d/2\) and stores it as a buffer.
  - Calls `_build_cache(max_seq_len)` to precompute `cos_cached` and `sin_cached` of shape `(max_seq_len, dim)`.
- **`_build_cache(seq_len)`**: Recomputes the frequency matrix and the corresponding cos/sin tables for a new maximum length. This is called automatically if the forward pass encounters a longer sequence.
- **`forward(x)`**:
  - Takes input of shape \((B, L, d)\).
  - If \(L > \text{max\_seq\_len}\), rebuilds cache.
  - Slices cos and sin tensors to length \(L\), adds batch and head dimensions.
  - Applies the rotation formula.
  - Returns the rotated tensor of the same shape.
- **`_rotate_half`**: Splits the last dimension in half, concatenates `[-second\_half, first\_half]`.
- **Integration**: The module is designed to be used as a building block within hybrid attention modules (e.g., Hymba‑TransX, SAMBA‑M3) to apply a unified RoPE to the hidden states before branching. It can also be registered as an attention replacement, though it does not perform attention itself; it merely transforms the input.

---

## 4. Usage and Configuration

The module can be injected via the framework by setting `attention_type` to `"compression_unified_rope"`. However, since it is a positional encoding layer rather than a full attention module, it is typically used as a sub‑component within other blocks. An example usage within a custom hybrid block:

```python
rope = UnifiedRoPE(dim=768, max_seq_len=4096)
x_rope = rope(x)
# then pass x_rope to both attention and SSM
```

Configuration parameters (passed via `comp_config` or constructor):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | required | Feature dimension (must be even). |
| `max_seq_len` | 4096 | Maximum sequence length for the cached rotation tables. |
| `base` | 10000 | Base for the inverse frequency computation. |

The module does not require pretrained weight transfer; it has no learnable parameters.

---

## 5. Discussion

UnifiedRoPE provides a clean separation of concerns: position encoding is abstracted into a single, reusable module that can be placed at the input of any hybrid block. Its precomputed tables ensure fast, deterministic application with minimal overhead. By applying the same rotation to all branches, it eliminates the risk of positional inconsistency that could arise when different branches use different position encoding schemes or when one branch lacks position information entirely.

One limitation is that the module expects the hidden dimension to be even and operates on the full feature vector. In architectures where the hidden state is split into heads with a different dimension, the rotation should be applied after the head split (as in standard RoPE). However, for hybrid blocks that operate on the full hidden state before splitting, UnifiedRoPE is perfectly suitable. The module also does not support arbitrary position offsets for incremental decoding; this could be added by passing an `offset` parameter, as some other RoPE implementations do.

The module is not a standalone attention mechanism; it is a helper component. Its registration in the `ATTENTION_REGISTRY` may be for convenience to allow injection via the same pipeline, but it must be used within a block that actually performs sequence mixing. The compression framework can treat it as a “pseudo‑attention” that transforms the input, and other modules can build on top of it.

---

## 6. Conclusion

We have presented UnifiedRoPE, a modular implementation of Rotary Position Embedding designed to unify positional encoding across multiple branches of hybrid Transformer architectures. The module precomputes and caches rotation tables, applies a consistent rotation to the input, and guarantees that all downstream components share the same positional representation. Its simple, parameter‑free design makes it a convenient building block for the compression framework. Future work could extend it with dynamic offset support and integrate it more tightly with the framework’s injection and training loops. UnifiedRoPE exemplifies the principle of modularity in efficient Transformer design.
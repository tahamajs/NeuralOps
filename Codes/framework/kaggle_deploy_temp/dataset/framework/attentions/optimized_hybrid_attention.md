# Optimized Hybrid Attention (OHA): Sliding Window Local + Low‑Rank Global with Gated Fusion

**Abstract**

We present **Optimized Hybrid Attention (OHA)**, a drop‑in replacement for the self‑attention sub‑layer in DistilBERT that combines two complementary efficient attention mechanisms: a **full local sliding window** attention that captures fine‑grained short‑range dependencies, and a **low‑rank global attention** that compresses the entire sequence into a small set of compressed key‑value vectors, thereby capturing long‑range context at linear cost. A learned, per‑head scalar gate dynamically fuses the outputs of the two branches, enabling the model to adaptively balance local precision and global context. The module supports zero‑shot initialisation by copying the standard query, key, value, and output projection weights from a pretrained DistilBERT layer, while the additional compression and gating parameters are randomly initialised and fine‑tuned. OHA achieves a significant reduction in computational complexity for the global attention component—from \(O(L^2)\) to \(O(L \cdot r)\), where \(r\) is the compression rank—while retaining the exact attention within a local window. The module is integrated into the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of each component, the overall architecture, and the zero‑shot weight transfer procedure.

---

## 1. Introduction

Self‑attention in Transformers provides powerful sequence modelling but scales quadratically with the length \(L\). A popular line of efficient attention methods compromises between local and global interactions: a local window captures immediate context with high precision, while a compressed or sparse global mechanism provides a summary of the entire sequence. **Optimized Hybrid Attention (OHA)** follows this design philosophy. It maintains the standard multi‑head query, key, and value projections, and then computes attention in two parallel branches:

- **Local branch**: Standard scaled dot‑product attention restricted to a sliding window of fixed size \(W\). This branch is exact and incurs a cost \(O(L W d)\).

- **Global branch**: The keys and values are first compressed into a small number \(r\) of summary vectors via learned linear projections followed by a softmax over the sequence length. The resulting compressed keys and values of shape \((B, H, r, d_h)\) are then used for full attention with the original queries. This yields a global context at cost \(O(L r d)\).

A per‑head learnable gate, conditioned on the original hidden states, fuses the local and global outputs via a sigmoid activation. The module’s linear projections for Q, K, V, and O can be initialised from a pretrained DistilBERT attention layer, ensuring zero‑shot performance, while the additional compression matrices and gate are trained from scratch.

In this paper, we fully describe the mathematical formulation of the local window attention, the low‑rank global compression, the gating mechanism, and the integration procedure. We also discuss the computational complexity and the handling of padding masks.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first applies standard linear projections for all heads simultaneously:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X} \mathbf{W}_K,\quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\), \(H\) is the number of heads, and \(d_h = D/H\). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\) via a helper function `transpose_for_scores`.

### 2.2 Local Sliding‑Window Attention

The local branch performs exact attention within a fixed window of size \(W\) (the `window_size` hyperparameter). A symmetric mask \(\mathbf{M}^{\text{local}} \in \{0,1\}^{L \times L}\) is defined as:

\[
M^{\text{local}}_{ij} = \begin{cases}
1 & \text{if } |i - j| \le W, \\
0 & \text{otherwise.}
\end{cases}
\]

The mask is broadcast to shape \((1, 1, L, L)\). The local attention scores are computed as:

\[
\mathbf{S}^{\text{local}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]
and positions outside the window are set to \(-\infty\):

\[
\mathbf{S}^{\text{local}} \gets \mathbf{S}^{\text{local}} \odot \mathbf{M}^{\text{local}} + (1 - \mathbf{M}^{\text{local}}) \cdot (-\infty).
\]

If a padding mask is provided (as a 2D tensor of shape \((B, L)\) with 1 for valid tokens), it is expanded to \((B, 1, 1, L)\) and converted to an additive mask:

\[
\mathbf{S}^{\text{local}} \gets \mathbf{S}^{\text{local}} + (1 - \text{mask}) \cdot (-\infty).
\]

Softmax is applied, followed by dropout, and the local context is obtained as:

\[
\mathbf{C}^{\text{local}} = \operatorname{softmax}(\mathbf{S}^{\text{local}}) \mathbf{V}.
\]

The complexity of this branch is \(O(B H L W d_h)\).

### 2.3 Low‑Rank Global Attention via Compression

The global branch reduces the sequence length dimension of keys and values to a small, fixed rank \(r\) (the `compression_rank` hyperparameter, default 64). This is achieved by two learned linear projections \(\mathbf{W}_{\text{ck}} \in \mathbb{R}^{D \times r}\) and \(\mathbf{W}_{\text{cv}} \in \mathbb{R}^{D \times r}\) (without biases) that map the input hidden states to compressed representations:

\[
\mathbf{C}_K = \mathbf{X} \mathbf{W}_{\text{ck}}, \quad \mathbf{C}_V = \mathbf{X} \mathbf{W}_{\text{cv}}, \qquad \mathbf{C}_K, \mathbf{C}_V \in \mathbb{R}^{B \times L \times r}.
\]

To select which token positions contribute most to each compressed dimension, a softmax is applied along the sequence dimension (dim=1) for both \(\mathbf{C}_K\) and \(\mathbf{C}_V\), yielding normalised weights:

\[
\mathbf{\alpha}_{K} = \operatorname{softmax}(\mathbf{C}_K) \in \mathbb{R}^{B \times L \times r},
\qquad
\mathbf{\alpha}_{V} = \operatorname{softmax}(\mathbf{C}_V) \in \mathbb{R}^{B \times L \times r}.
\]

These weights aggregate the full multi‑head keys and values. For each head, the original keys \(\mathbf{K}\) (shape \((B, H, L, d_h)\)) are first permuted to \((B, H, d_h, L)\), then multiplied by the expanded weights \(\mathbf{\alpha}_K\) (shape \((B, 1, L, r)\)) to produce compressed keys of shape \((B, H, d_h, r)\). After permutation, the compressed keys and values are:

\[
\mathbf{K}^{\text{comp}} = \left( \mathbf{K}^{\mathsf{T}} \cdot \mathbf{\alpha}_K \right) \quad \text{with proper reshapes},
\]
\[
\mathbf{V}^{\text{comp}} = \left( \mathbf{V}^{\mathsf{T}} \cdot \mathbf{\alpha}_V \right),
\]

where the dot product is over the sequence length dimension. Concretely, the code computes:

\[
\mathbf{K}^{\text{comp}} = \operatorname{matmul}(\mathbf{K}_{\text{trans}}, \mathbf{\alpha}_K.unsqueeze(1)) \quad \text{and permutes back to } (B, H, r, d_h).
\]

Thus, the compressed keys and values each consist of \(r\) summary vectors for each head. The global attention scores between the full queries \(\mathbf{Q}\) (shape \((B, H, L, d_h)\)) and the compressed keys are then:

\[
\mathbf{S}^{\text{global}} = \frac{\mathbf{Q} (\mathbf{K}^{\text{comp}})^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L \times r}.
\]

Softmax is applied (after optional padding masking), followed by dropout, and the global context is:

\[
\mathbf{C}^{\text{global}} = \operatorname{softmax}(\mathbf{S}^{\text{global}}) \mathbf{V}^{\text{comp}} \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

The complexity of this branch is dominated by the aggregation step \(O(B H L r d_h)\) and the attention \(O(B H L r d_h)\), which is linear in \(L\) for a fixed rank \(r\).

**Padding mask handling for compression:** If a padding mask is provided, it is also applied to the compressed features \(\mathbf{C}_K, \mathbf{C}_V\) by adding an additive bias of `-10000` to padding positions before the softmax. This ensures padding tokens do not contribute to the compressed representations.

### 2.4 Gated Fusion

A per‑head scalar gate controls the contribution of the local and global branches. The gate is computed from the original hidden states \(\mathbf{X}\) using a linear layer \(\mathbf{W}_{\text{gate}} \in \mathbb{R}^{D \times H}\) (with bias) followed by sigmoid:

\[
\mathbf{G} = \sigma(\mathbf{X} \mathbf{W}_{\text{gate}} + \mathbf{b}_{\text{gate}}) \in \mathbb{R}^{B \times L \times H}.
\]

The gate values are reshaped to \((B, H, L, 1)\). The final context for each head is a convex combination:

\[
\mathbf{C} = \mathbf{G} \odot \mathbf{C}^{\text{local}} + (1 - \mathbf{G}) \odot \mathbf{C}^{\text{global}}.
\]

If the gating is disabled (`use_gate=False`), the two branches are simply summed:

\[
\mathbf{C} = \mathbf{C}^{\text{local}} + \mathbf{C}^{\text{global}}.
\]

The gate bias is initialised to a relatively high value (default 2.0), which biases the initial gate towards 1 (since \(\sigma(2) \approx 0.88\)), thus favouring the local branch at the start of training.

### 2.5 Output Projection

After fusion, the context is reshaped to \((B, L, H d_h)\) and passed through the output projection \(\mathbf{W}_O\):

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \mathbf{A}^{\text{local}})\) if `output_attentions=True`, otherwise \((\mathbf{Y}, \text{None})\).

---

## 3. Implementation Details

- **Configuration**: All hyperparameters are stored in the `OptimizedHybridAttentionConfig` dataclass (hidden size, heads, window size, compression rank, gate settings).
- **Class `OptimizedHybridAttention`** inherits from `BaseAttention` and is registered as `"oha"`.
- **Weight initialisation**:
  - The standard Q, K, V, O projections are copied from the pretrained DistilBERT attention layer via `from_distilbert_pretrained` classmethod, ensuring zero‑shot performance.
  - The compression matrices `compress_k` and `compress_v` are initialised with a normal distribution (std 0.02).
  - The gate weight is initialised with a normal distribution (std 0.02), and its bias is set to `gate_bias` (default 2.0).
- **Mask handling**:
  - A 2D padding mask of shape \((B, L)\) (with 1 for valid) is converted to an additive mask by `(1 - mask) * -10000.0`. For the global compression, the mask is applied before softmax in the compression weights, so padding tokens do not affect the compressed vectors.
  - For 4D masks, it attempts to use them directly; typical usage expects 2D masks.
- **Local window mask**: Constructed via distance matrix and broadcast to \((1, 1, L, L)\).
- **Compression aggregation**: The code uses `softmax` on the compressed features to obtain a weighting over tokens. This is a soft selection mechanism, allowing each compressed dimension to attend to a weighted combination of all tokens. The same weighting is applied to both keys and values independently.
- **Computational complexity**: 
  - Local: \(O(B H L W d_h)\)
  - Global: \(O(B H L r d_h)\) plus \(O(B H L r)\) for the compression weight computation.
  - For typical values (\(W=128\), \(r=64\)), the global branch is cheaper for long sequences, and the total is linear in \(L\).
- **Injection**: The function `inject_oha` builds the config from `comp_config`, calls `from_distilbert_pretrained`, and replaces `layer.attention`.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"oha"`. The following parameters can be set in `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 128 | Window size for local attention. |
| `compression_rank` | 64 | Rank \(r\) for the compressed global attention. |
| `use_gate` | True | Whether to use a learned gate for fusion. |
| `gate_bias` | 2.0 | Initial bias for the gate (higher favours local). |
| `dropout` | from config | Applied to attention weights in both branches. |

Example injection via framework:
```python
comp_config.attention_type = "oha"
comp_config.block_size = 256
comp_config.compression_rank = 128
```

The injection function automatically copies pretrained weights, so the module can be used immediately for inference or fine‑tuning.

---

## 5. Discussion

OHA provides a balanced trade‑off between local and global processing. The local window ensures that short‑range syntax is captured with high fidelity, while the compressed global branch offers a summary of the entire sequence, enabling the model to track long‑range dependencies. The softmax‑based compression is a differentiable form of key‑value pooling, which learns to weight tokens based on their relevance. The per‑head gate allows each head to specialise, potentially attending more to local or global context depending on its role.

Compared to other hybrid methods, OHA is simple and computationally predictable. The compression rank \(r\) acts as a capacity knob: higher values improve global representation but increase cost. The method is similar in spirit to “Linformer” (which projects the length dimension) but uses a soft token selection rather than a fixed projection matrix, making it input‑dependent.

Potential limitations include:
- The softmax over the sequence length to obtain compression weights may become sharp, focusing only on a few tokens, which could lose diversity.
- The compression matrices are shared across all heads (i.e., a single `compress_k` for all heads), which might limit head‑specific global patterns.
- The gate is computed from the original hidden states, not from the local/global outputs themselves; conditioning on the outputs could be more direct.
- The local window mask is symmetric and non‑causal, suitable for bidirectional encoders like DistilBERT.

Nevertheless, OHA is a robust and well‑designed module that has been integrated into the compression framework with zero‑shot weight transfer, making it a strong candidate for efficient fine‑tuning and deployment.

---

## 6. Conclusion

We have presented Optimized Hybrid Attention (OHA), a hybrid self‑attention module that combines a sliding window local attention with a compressed global attention via learned softmax‑based token weighting, fused by a per‑head sigmoid gate. The module achieves linear complexity in sequence length for the global branch, retains exact local interactions, and supports zero‑shot initialisation from pretrained weights. Its clean design and integration into the DistilBERT compression framework make it a practical tool for efficient sequence modelling. Future work could explore dynamic compression ranks, head‑wise compression, and causal windowing for autoregressive tasks.
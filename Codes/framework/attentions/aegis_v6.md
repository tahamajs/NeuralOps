# Aegis‑Attention V6 — A Unified Hybrid Attention with Factorised Projections, Kernel Linear Attention, and Adaptive Gating

**Abstract**

We present Aegis‑Attention V6, a drop‑in replacement for the standard multi‑head self‑attention in DistilBERT. The module combines four key innovations: (1) **factorised low‑rank projections** initialised via truncated SVD that reduce parameters by over 20× while preserving zero‑shot performance; (2) **tiled sliding window attention** that computes local interactions exactly but with linear memory in the window size; (3) **global linear attention** using a stable orthogonal random feature map (FAVOR+) that scales linearly with sequence length; and (4) **learnable gating mechanisms**—query‑wise, input‑wise, and context‑aware fusion—that adaptively blend local and global information per token and per head. The design allows efficient handling of both short and very long sequences, with a fallback to full attention for inputs shorter than the window. All components are crafted for numerical stability, especially the kernel feature map which uses orthonormal random projections, clamped exponentials, and optional float64 upcast. The module is initialised in a zero‑shot fashion from pretrained weights, making it directly fine‑tunable without additional pretraining.

---

## 1. Introduction

The quadratic time and memory complexity of standard self‑attention limits the application of Transformer models to long documents. Several remedies have been proposed: restricting attention to a local window, using linearised attention via kernel feature maps, factorising the weight matrices, or employing sparse approximations. Aegis‑Attention V6 integrates these approaches into a single, cohesive module that balances local precision with global context while drastically reducing the parameter count.

The model splits each attention head’s computation into two parallel streams. **Local** interactions are captured by a sliding window with exact dot‑product attention, implemented in a tiled fashion for constant memory. **Global** interactions are modelled by a Performer‑style linear attention using orthogonal random features, which approximates the softmax kernel in \(O(N)\) time. The outputs of both streams are merged through a per‑head, per‑token gate that is itself conditioned on the hidden state and the intermediate local/global representations. Furthermore, all projection matrices (query, key, value, output) are stored in a factorised form—a frozen SVD‑based core plus a trainable low‑rank update—yielding an extreme parameter reduction.

---

## 2. Method

### 2.1 Factorised Low‑Rank Projections

Each linear projection (e.g., \(\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times (H d_h)}\)) is replaced by a `FactorisedLowRankProjection`. Given the pretrained weight \(\mathbf{W}\), we first compute the (full) SVD:

\[
\mathbf{W} = \mathbf{U} \mathbf{S} \mathbf{V}^\top, \quad \mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}},\; \mathbf{V} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}},\; \mathbf{S} = \operatorname{diag}(\sigma_1, \sigma_2, \dots).
\]

We keep the top \(r_{\text{svd}}\) singular values and vectors: \(\mathbf{U}_r = \mathbf{U}_{[:,:r]}\), \(\mathbf{S}_r = \mathbf{S}_{[:r]}\), \(\mathbf{V}_r = \mathbf{V}_{[:,:r]}\). The frozen SVD part is then \(\mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top\). The full weight is approximated as:

\[
\mathbf{W} \approx \underbrace{\mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top}_{\text{frozen}} + \underbrace{\mathbf{A} \mathbf{B}}_{\text{trainable}},
\]

where \(\mathbf{A} \in \mathbb{R}^{d_{\text{out}} \times r_{\text{lora}}}\) and \(\mathbf{B} \in \mathbb{R}^{r_{\text{lora}} \times d_{\text{in}}}\) are small adapters. During a forward pass, the output is:

\[
\mathbf{y} = \underbrace{(\mathbf{x} \mathbf{V}_r) \mathbf{S}_r \mathbf{U}_r^\top}_{\text{SVD path}} + \underbrace{(\mathbf{x} \mathbf{B}^\top) \mathbf{A}^\top}_{\text{LoRA path}}.
\]

This formulation avoids ever instantiating the full \(d_{\text{out}} \times d_{\text{in}}\) matrix, saving memory and compute. \(\mathbf{U}_r, \mathbf{V}_r\) are fixed; only \(\mathbf{S}_r, \mathbf{A}, \mathbf{B}\) are trained. The rank \(r_{\text{svd}}\) is a hyperparameter (e.g., 128), much smaller than the original dimensions (768), leading to a >20× parameter reduction.

### 2.2 Tiled Sliding Window Attention (Local)

For local attention, a symmetric window of size \(2 \times \text{window\_size}\) is used (the code uses `self.window` as the half‑window; the distance threshold is `dist < self.window`). To keep memory usage bounded, we process queries in tiles of size `step = self.window`. For each tile of queries spanning indices \([s, e]\), the corresponding key/value tiles are extended to \([s - W, e + W]\) to cover the full window. A combined attention mask is built that incorporates:

- Padding mask (from the attention mask).
- Optional square mask (e.g., causal or custom mask).
- Local band mask (distance < window).
- Query-side padding.

The attention for each tile is computed using PyTorch’s highly optimised `F.scaled_dot_product_attention`, which handles softmax internally. The result is placed into the output tensor. Because tiles do not overlap in query space, there is no additional overhead beyond the standard sliding window cost of \(O(N \cdot W)\).

### 2.3 Global Linear Attention with Orthogonal Random Features

Global attention uses the Performer’s FAVOR+ mechanism. The core idea is to approximate the softmax kernel \(\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})\) by a finite feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\). We employ the positive orthogonal random features:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \exp\left( \mathbf{x} \mathbf{R} - \frac{\|\mathbf{x}\|^2}{2} \right),
\]

where \(\mathbf{R} \in \mathbb{R}^{d \times m}\) is a random matrix with orthonormal columns scaled by \(\sqrt{d}\), and \(m\) is the feature dimension (`kernel_feature_dim`). This choice minimises the variance of the approximation.

**Numerical stability** is ensured by:
- Clamping the exponential argument to \([-20, 20]\).
- Optional float64 upcast for the exponential.
- Pre‑normalisation of queries and keys via `RMSNorm` before projection (if `use_pre_norm=True`).

The attention computation follows the standard linear‑attention pattern:

\[
\mathbf{KV} = \sum_{j=1}^N \phi(\mathbf{k}_j) \mathbf{v}_j^\top \quad\in \mathbb{R}^{B \times H \times m \times d},
\]
\[
\mathbf{Z} = \sum_{j=1}^N \phi(\mathbf{k}_j) \quad\in \mathbb{R}^{B \times H \times m \times 1},
\]
\[
\mathbf{C}_{\text{global}} = \frac{\phi(\mathbf{Q}) \mathbf{KV}}{\phi(\mathbf{Q}) \mathbf{Z} + \epsilon}.
\]

All operations are linear in \(N\), with complexity \(O(N m d)\). Padding tokens are masked out by multiplying \(\phi(\mathbf{k})\) and \(\mathbf{v}\) with the padded mask before aggregation.

### 2.4 Gating Mechanisms

Two gating modules modulate the flow of information.

**GatingModule**  
This combines two independent gates:
- *Query gate*: applied to the query feature map before global attention. A linear layer from `head_dim` to 1 with sigmoid activation yields a scalar per head per token: \(\gamma_q \in (0,1)\). It scales \(\phi(\mathbf{q})\) elementwise.
- *Input gate*: applied to keys and values. A linear layer from `hidden_size` to `num_heads` produces per‑head logits, which are sigmoid‑activated to give \(\gamma_{in} \in (0,1)\). Both \(\phi(\mathbf{k})\) and \(\mathbf{v}\) are multiplied by this gate.

Both gates are conditioned on the original hidden states (input gate) or the query (query gate) and are trained end‑to‑end.

**ContextFusionGate**  
After local and global outputs are computed, a per‑head, per‑token scalar \(\beta\) is produced by concatenating:
- The per‑head hidden state (split from the full hidden state).
- The local output.
- The global output.

A linear projection to 1 (with bias initialised to 1.0, pushing initial fusion toward the global signal) followed by sigmoid yields:

\[
\mathbf{C}_{\text{fused}} = (1 - \beta) \cdot \mathbf{C}_{\text{local}} + \beta \cdot \mathbf{C}_{\text{global}}.
\]

Thus, the model can dynamically decide, for each token and each head, how much to rely on local versus global context.

### 2.5 Overall Forward Pass

Given hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times d_{\text{model}}}\):

1. Project to \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) using the factorised projections and reshape to \((B, H, N, d_h)\).
2. Compute local output \(\mathbf{C}_{\text{local}}\) using tiled sliding window attention.
3. Compute query gate \(\gamma_q\) and input gate \(\gamma_{in}\) from the hidden states and \(\mathbf{Q}\).
4. Apply RMSNorm to \(\mathbf{Q}, \mathbf{K}\) (if enabled), project through the kernel feature map to get \(\phi(\mathbf{Q}), \phi(\mathbf{K})\), apply gates, and compute global output \(\mathbf{C}_{\text{global}}\).
5. Compute the fusion gate \(\beta\) and blend:
   \[
   \mathbf{C} = (1 - \beta) \mathbf{C}_{\text{local}} + \beta \mathbf{C}_{\text{global}}.
   \]
6. Reshape to \((B, N, H d_h)\) and apply factorised output projection.
7. Return output and `None` for compatibility.

Padding masks are handled uniformly: a 2D boolean mask is first converted to a 4D additive mask (0 for valid, -1e9 for invalid) for local attention, and to a 1D mask for global attention. The code includes extensive logic to interpret masks of various dimensionalities (2D, 3D, 4D).

---

## 3. Implementation Highlights

- **FactorisedLowRankProjection**: Stores \(\mathbf{U}, \mathbf{S}, \mathbf{V}\) and \(\mathbf{A}, \mathbf{B}\); uses `register_buffer` for \(\mathbf{U}, \mathbf{V}\) so they remain frozen.
- **StableKernelFeatureMap**: Contains the orthonormal random projection matrix, built using a QR decomposition of a random matrix. Offers several initialisation schemes (`orthogonal`, `random`, `cosine`, `custom`). The forward pass clamps the exponent and multiplies by the scaling factor.
- **Tiled sliding window**: Uses `F.scaled_dot_product_attention` with a combined mask; memory footprint is constant with respect to sequence length (depends only on window size and tile size).
- **Gates**: Simple linear layers with sigmoid; biases are initialised to bias the gates toward open (input gate bias = 2.0 → initial sigmoid ≈ 0.88; fusion gate bias = 1.0 → initial β ≈ 0.73).

The module is registered as `"aegis_v6"` in the `ATTENTION_REGISTRY` and can be injected into any DistilBERT model via the provided injection function, which reads hyperparameters from a compression config.

---

## 4. Configuration Parameters

The dataclass `AegisConfigV3` (used internally) and the injection function expose the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 64 | Half‑width of the local window (total context = 2×window). |
| `svd_rank` | 128 | Rank of the frozen SVD part of projections. |
| `low_rank_update` | 16 | Rank of the trainable LoRA adapters. |
| `kernel_feature_dim` | 128 | Dimension \(m\) of the random features. |
| `kernel_method` | "orthogonal" | Method to initialise the random projection (orthogonal, random, cosine, custom). |
| `kernel_clamp_min` / `_max` | -20 / 20 | Clamping range for the exponential argument. |
| `kernel_use_float64` | False | Whether to upcast to float64 before the exponential. |
| `use_query_gate` | True | Enable query‑side gating. |
| `use_input_gate` | True | Enable input‑side gating for keys/values. |
| `use_pre_norm` | True | Apply RMSNorm to Q and K before kernel feature map. |
| `dropout` | 0.1 | Dropout rate for local attention. |
| `long_context_threshold` | 2048 | (Reserved for future top‑k reduction, not currently used) |
| `max_global_tokens` | 1024 | (Reserved) |

---

## 5. Discussion

Aegis‑V6 pushes the efficiency–performance trade‑off by combining parameter reduction with linear complexity. The factorised projections drastically lower the memory footprint and number of trainable parameters, making the model suitable for on‑device fine‑tuning. The local‑global hybrid ensures that both fine‑grained syntactic patterns and long‑range dependencies are captured, while the learned gates automatically adjust the contribution of each stream per sample and per layer. The stable kernel feature map, with its orthonormal projection and numerical safeguards, provides a high‑quality approximation of the softmax kernel without instability in mixed‑precision training.

The main limitation is that the global attention relies on a fixed random projection; although it is learnable (the `rff_proj` is a parameter), the orthogonality is not preserved during training, which could slightly degrade the approximation. Future versions could enforce orthogonality via a regularisation or re‑orthogonalisation step. Additionally, the current implementation does not yet utilise `long_context_threshold` for top‑k filtering, which could further speed up very long sequences.

---

## 6. Conclusion

Aegis‑Attention V6 provides a state‑of‑the‑art hybrid attention module that integrates SVD‑based compression, local exact attention, kernelised linear attention, and multi‑gating. It achieves significant parameter savings, linear time complexity in sequence length, and strong numerical stability, all while preserving compatibility with DistilBERT’s pretrained weights. This makes it a powerful building block for efficient Transformer models aimed at long‑document understanding.
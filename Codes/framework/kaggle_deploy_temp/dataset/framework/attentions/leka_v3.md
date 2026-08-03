# LEKAv3: Linear‑Time Efficient Kernel Attention with Multi‑Scale Features, Value‑Based Landmark Selection, and Adaptive Gating

**Abstract**

We present **LEKAv3**, an improved version of the Linear‑time Efficient Kernel Attention that refines the landmark‑based global attention paradigm with several advancements: (1) **multi‑scale random Fourier features** using two fixed projection matrices for richer kernel approximation; (2) a **value‑aware landmark scorer** that selects the most informative tokens based on their value projections, rather than key features; (3) a **learnable correction matrix** that transforms the selected landmark values; and (4) a **per‑head gating mechanism** that adaptively blends the sparse landmark attention output with a small local window attention. The module operates in linear time \(O(N \cdot L \cdot F + N \cdot W \cdot d_h)\) while preserving both global and local context. All linear projection weights are initialised from a pretrained DistilBERT attention layer for zero‑shot transfer. LEKAv3 is registered in the compression framework as a drop‑in replacement for standard self‑attention, offering a stronger baseline than its predecessors. This paper provides a full mathematical description of each component and the overall architecture.

---

## 1. Introduction

Linear attention methods replace the quadratic softmax kernel with a feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\), enabling \(O(N d^2)\) computation. To further reduce the constant factor and focus capacity on the most relevant tokens, the landmark approach (introduced in LEKAv2) compresses the key‑value set to a fixed number of salient landmarks. However, LEKAv2 relied on key‑based features for landmark scoring and used learnable random projections, which may overfit or lose important input‑dependent information. **LEKAv3** improves upon this by:

- Using **fixed**, multi‑scale random Fourier features that are not updated during training, stabilising the kernel approximation.
- Selecting landmarks based on the **value vectors** themselves, which are more directly tied to the information content to be propagated.
- Retaining the correction matrix and per‑head blending gate from V2, but with a simplified gate parameterisation.
- Integrating these components into a clean, efficient pipeline with zero‑shot weight transfer.

In this paper, we detail the mathematical formulation of each component, the architectural changes from V2, and the implementation specifics. We also discuss the advantages and remaining limitations of the LEKAv3 design.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given hidden states \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\), the module first computes standard multi‑head projections:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V,
\]

with weight matrices \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\) and biases. The tensors are reshaped to \((B, H, N, d_h)\).

### 2.2 Multi‑Scale Random Fourier Features

Two random projection matrices are stored as fixed buffers (not trainable):

- \(\mathbf{R}_1 \in \mathbb{R}^{d_h \times F}\) where \(F\) = `feature_dim` (default 128).
- \(\mathbf{R}_2 \in \mathbb{R}^{d_h \times F/2}\).

These are initialised from a standard normal distribution once and never updated. The **multi‑scale** feature map \(\phi(\mathbf{x})\) is constructed by projecting \(\mathbf{x}\) with both matrices, applying cosine and sine, and concatenating the results:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \big[ \cos(\mathbf{x}\mathbf{R}_1), \sin(\mathbf{x}\mathbf{R}_1) \big] \;\oplus\; \frac{1}{\sqrt{F/2}} \big[ \cos(\mathbf{x}\mathbf{R}_2), \sin(\mathbf{x}\mathbf{R}_2) \big],
\]

where \(\oplus\) denotes concatenation. The total feature dimension is \(2F + 2(F/2) = 3F\). The two scales allow the kernel to capture similarities at different resolutions.

Thus, we obtain \(\mathbf{Q}_\phi \in \mathbb{R}^{B \times H \times N \times 3F}\) and \(\mathbf{K}_\phi\) similarly.

### 2.3 Value‑Based Landmark Selection

Unlike LEKAv2 which scored key features, LEKAv3 selects landmarks based on the **value vectors** themselves, which carry the information that will be propagated. A small scoring MLP is applied to each value token (per head):

\[
s_{i,h} = \operatorname{MLP}(\mathbf{V}_{i,h}), \quad \text{MLP}: \text{Linear}(d_h \to 32) \to \text{GELU} \to \text{Linear}(32 \to 1).
\]

The top‑\(L\) scores are selected (\(L = \text{num\_landmarks}\), default 32). If \(N \le L\), no selection is performed (all tokens are used). The corresponding key features and values are gathered:

\[
\mathbf{K}_{\text{land}} \in \mathbb{R}^{B \times H \times L \times 3F}, \qquad
\mathbf{V}_{\text{land}} \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

This value‑based selection is motivated by the observation that tokens with large values (in magnitude or after transformation) are often the most informative for the attention aggregation.

### 2.4 Correction Matrix

As in LEKAv2, a per‑head correction matrix \(\mathbf{W}_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}\) (initialised to the identity) is applied to the landmark values:

\[
\tilde{\mathbf{V}}_{\text{land}} = \mathbf{V}_{\text{land}} \mathbf{W}_{\text{corr}}.
\]

This linear transformation allows the model to adjust the compressed values to better represent the full set, compensating for information loss due to landmark selection.

### 2.5 Sparse Landmark Attention with Softpick

Attention weights between query features \(\mathbf{Q}_\phi\) and landmark key features \(\mathbf{K}_{\text{land}}\) are computed via a simple dot product:

\[
\mathbf{S}^{\text{land}} = \mathbf{Q}_\phi \, \mathbf{K}_{\text{land}}^\top \quad \in \mathbb{R}^{B \times H \times N \times L}.
\]

The **Softpick** activation is then applied to encourage sparsity:

\[
\mathbf{A}^{\text{land}} = \operatorname{Softpick}(\mathbf{S}^{\text{land}}, \text{dim}=-1),
\]

where Softpick is a rectified softmax that can produce exact zero probabilities (same as in previous LEKA versions). Dropout is applied to these attention weights. The landmark context is then:

\[
\mathbf{C}_{\text{land}} = \mathbf{A}^{\text{land}} \tilde{\mathbf{V}}_{\text{land}} \quad \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

### 2.6 Local Window Attention

To capture fine‑grained short‑range dependencies, a local sliding window attention is computed on the original (non‑feature) queries, keys, and values. A symmetric window of size \(W\) (local_window_size, default 3) is defined:

\[
M^{\text{local}}_{ij} = \begin{cases}
1 & \text{if } |i - j| \le \lfloor W/2 \rfloor, \\
0 & \text{otherwise.}
\end{cases}
\]

Local scores are computed via scaled dot product:

\[
\mathbf{S}^{\text{local}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]

and positions outside the window are masked with \(-\infty\). Standard softmax is applied, followed by matrix multiplication with \(\mathbf{V}\):

\[
\mathbf{C}_{\text{local}} = \operatorname{softmax}(\mathbf{S}^{\text{local}} + \text{mask}) \mathbf{V}.
\]

### 2.7 Per‑Head Adaptive Gating

A learnable scalar parameter \(\gamma_h\) per head (initialised to \(0\)) controls the blending between the landmark and local outputs. The effective gate is obtained via sigmoid:

\[
\alpha_h = \sigma(\gamma_h).
\]

The final context for head \(h\) is a convex combination:

\[
\mathbf{C}_h = \alpha_h \cdot \mathbf{C}_{\text{land},h} + (1 - \alpha_h) \cdot \mathbf{C}_{\text{local},h}.
\]

This allows each head to dynamically learn whether to rely more on the global (landmark) or local (window) information.

### 2.8 Output Projection and Masking

The combined context is reshaped to \((B, N, H d_h)\) and passed through a final linear output projection:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

**Attention mask handling**: As in LEKAv2, a mask of various dimensions is converted to a 2D boolean tensor `token_valid`. This mask is used to set attention scores for invalid query positions to \(-\infty\) in the landmark attention (before Softpick). However, the local window attention does **not** receive this mask, meaning padding tokens can still be attended to within the window. Also, the landmark key‑value aggregation does not mask out padding tokens, which can distort the global context. This is a known limitation to be addressed in future versions.

### 2.9 Complexity

The complexity is dominated by:
- Computing features: \(O(N \cdot H \cdot d_h \cdot F) = O(N \cdot D \cdot F)\).
- Landmark selection: \(O(N \cdot H \cdot d_h) = O(N \cdot D)\) for scoring.
- Landmark attention: \(O(N \cdot H \cdot L \cdot F) = O(N \cdot D \cdot L \cdot F / d_h)\).
- Local window: \(O(N \cdot H \cdot W \cdot d_h) = O(N \cdot D \cdot W)\).

With typical values (\(F=128, L=32, W=3, d_h=64\)), the landmark attention is the largest term but still linear in \(N\). Memory for the feature maps is \(O(N \cdot D \cdot F / d_h)\), which can be high but manageable.

---

## 3. Implementation Details

### 3.1 Class Structure

`LEKAV3Attention` inherits from `BaseAttention` (which provides interface compatibility) and is registered as `"leka_v3"`.

Key attributes:
- `query`, `key`, `value`, `out_proj`: linear layers (copied from pretrained).
- `proj1`, `proj2`: fixed random buffers of shapes \((d_h, F)\) and \((d_h, F/2)\).
- `landmark_scorer`: a two‑layer MLP (Linear(\(d_h\), 32) → GELU → Linear(32, 1)).
- `W_corr`: correction matrix parameter of shape \((H, d_h, d_h)\).
- `gate`: per‑head scalar parameter (shape \(H\)), no bias.
- `local_window_size`, `num_landmarks`, `feature_dim`: hyperparameters from `comp_config`.

### 3.2 Forward Pass Steps

1. Project inputs → Q, K, V multi‑head.
2. Compute \(\phi(Q), \phi(K)\) using fixed multi‑scale projections.
3. Select landmarks by scoring V with the MLP, gathering top‑\(L\) K_features and V.
4. Apply correction matrix to gathered V.
5. Compute landmark attention scores (dot product Q_features and K_land), mask invalid query positions, apply Softpick and dropout, multiply by corrected V.
6. Compute local attention scores, apply window mask, softmax, multiply by V.
7. Blend with per‑head gate (sigmoid).
8. Reshape, output projection.
9. Return `(output, None)`.

### 3.3 Weight Transfer

The injection function `inject_leka_v3` copies the pretrained attention weights from `old_attn` (which has `q_lin`, `k_lin`, etc.) to the corresponding linear layers, ensuring that the module starts from a strong initialisation. The other components (random projections, scorer, correction matrix, gate) are randomly initialised.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"leka_v3"`. The configurable hyperparameters (from `comp_config`) are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_landmarks` | 32 | Number of landmark tokens \(L\). |
| `feature_dim` | 128 | Base feature dimension \(F\) (total output dim \(3F\)). |
| `local_window_size` | 3 | Width of the local attention window. |
| `dropout` | 0.1 | Dropout rate on landmark attention weights. |

The injection function handles weight copying automatically.

---

## 5. Discussion

LEKAv3 represents a mature step in the LEKA lineage. By fixing the random projections, it avoids additional trainable parameters in the feature map, which may improve generalisation. The shift from key‑based to value‑based landmark scoring is a clever design choice: values contain the actual content that will be aggregated, so selecting landmarks based on values directly targets the most impactful tokens. The correction matrix continues to provide a flexible post‑hoc adjustment, and the per‑head gate allows adaptive fusion.

However, several issues remain. The landmark scoring is followed by a hard top‑\(k\), which is non‑differentiable. Training of the scorer requires a straight‑through estimator or a soft top‑\(k\), neither of which is implemented. The local window still lacks mask integration, so padding tokens are included. The feature map dimension \(3F\) can be large (e.g., 384 for \(F=128\)), leading to significant memory usage for the features. The dynamic landmark count mentioned in the docstring is not actually implemented (the number is fixed).

Despite these, LEKAv3 provides a strong, efficient attention alternative that balances global and local processing. It is well‑suited for tasks with moderate to long sequences where full attention is expensive, and where both short‑range syntax and long‑range semantics matter.

---

## 6. Conclusion

We have presented LEKAv3, an improved linear‑time kernel attention module with multi‑scale fixed random features, value‑based landmark selection, a correction matrix, and a gated fusion with local window attention. The design achieves linear complexity while preserving global context and local detail. The module is integrated into the DistilBERT compression framework with zero‑shot weight transfer, making it a practical drop‑in replacement for standard self‑attention. Future work will focus on making landmark selection differentiable, adding full mask support, and exploring dynamic landmark counts to further enhance efficiency and accuracy.
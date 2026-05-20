# LEKAv2: Linear‑Time Efficient Kernel Attention with Landmark Selection and Local Window Gate

**Abstract**

We present **LEKAv2**, an advanced linear‑time attention module that builds upon the random Fourier feature approach of LEKAv1, introducing three key innovations: (1) **landmark selection** to compress the key‑value memory into a small set of salient tokens, (2) a **correction matrix** that refines the landmarks via a learned linear transformation, and (3) a **per‑head gated fusion** between the landmark‑based global attention and a small local window attention using softpick. The module employs dual‑scale random features for richer kernel approximation and uses a novel softpick activation for sparse landmark attention. All linear projections are initialised from a pretrained DistilBERT attention layer for zero‑shot transfer. LEKAv2 achieves linear complexity in sequence length while retaining the ability to capture long‑range dependencies through landmarks and precise local interactions through the window. The module is registered in the compression framework as a drop‑in replacement for standard self‑attention. This paper provides a rigorous mathematical description of each component and the overall architecture.

---

## 1. Introduction

Linear attention methods, such as LEKAv1, replace the quadratic softmax attention with kernel feature maps, enabling \(O(N d^2)\) computation. However, they still aggregate all key‑value pairs into a single memory, which can be costly if the feature dimension \(F\) is large and may dilute the influence of important tokens. The **landmark** approach (e.g., Nyströmformer, Performer‑landmarks) selects a small subset of tokens to serve as a reduced key‑value set, drastically cutting the memory size. **LEKAv2** adopts this idea: it learns to select a fixed number of landmarks per head, computes attention to those landmarks using random Fourier features and a softpick activation, and then refines the values via a learned correction matrix. In parallel, a small **local window attention** captures fine‑grained short‑range interactions. The two outputs are blended per head via a learned gate. This design achieves \(O(N L F + N W d)\) complexity, where \(L\) is the number of landmarks (e.g., 16) and \(W\) the window size (e.g., 5), remaining linear in \(N\) while preserving global and local information.

In this paper, we detail the full mathematical pipeline of LEKAv2, including the dual‑scale random features, landmark scoring, correction matrix, softpick activation, local window attention, and adaptive gating. We also describe the injection mechanism with zero‑shot weight transfer.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), the module projects to queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to \((B, H, N, d_h)\).

### 2.2 Dual‑Scale Random Fourier Features

Two learnable random projection matrices are used (unlike fixed projections in LEKAv1):

- \(\mathbf{R}_1 \in \mathbb{R}^{d_h \times F}\) (with \(F = \text{feature\_dim}\))
- \(\mathbf{R}_2 \in \mathbb{R}^{d_h \times F/2}\)

These are `nn.Parameter` initialised with small random values (\(\sigma=0.02\)). The feature map \(\phi(\mathbf{x})\) is constructed by concatenating the cosine and sine of the projections at both scales:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \big[ \cos(\mathbf{x}\mathbf{R}_1), \sin(\mathbf{x}\mathbf{R}_1) \big] \;\oplus\; \frac{1}{\sqrt{F/2}} \big[ \cos(\mathbf{x}\mathbf{R}_2), \sin(\mathbf{x}\mathbf{R}_2) \big],
\]

where \(\oplus\) denotes concatenation. The total feature dimension is \(2F + 2(F/2) = 3F\). This dual‑scale design allows the model to capture both fine and coarse kernel similarities.

Thus, we obtain \(\mathbf{Q}_\phi \in \mathbb{R}^{B \times H \times N \times 3F}\) and \(\mathbf{K}_\phi\) similarly.

### 2.3 Landmark Selection

Instead of aggregating all \(N\) keys, a subset of \(L\) landmarks is selected dynamically for each head. A small **landmark scorer** network computes a score for each key feature:

\[
s_{i,h} = \text{Scorer}(\mathbf{K}_{\phi,i,h}) \in \mathbb{R},
\]

where the scorer is an MLP: Linear(\(3F \to F/2\)) → GELU → Linear(\(F/2 \to 1\)). The top‑\(L\) scores are chosen, and the corresponding key features and values are gathered:

\[
\mathbf{K}_{\text{land}} \in \mathbb{R}^{B \times H \times L \times 3F}, \quad
\mathbf{V}_{\text{land}} \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

If \(N \le L\), the full set is used without selection.

### 2.4 Correction Matrix

A learnable correction matrix \(\mathbf{W}_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}\) is applied to the landmark values to adjust them before attention. It is initialised to the identity matrix for each head. The corrected values are:

\[
\tilde{\mathbf{V}}_{\text{land}} = \mathbf{V}_{\text{land}} \mathbf{W}_{\text{corr}} \quad \text{(using batch matrix multiply)}.
\]

This linear transformation allows the landmarks to better represent the original values after compression.

### 2.5 Softpick Landmark Attention

Attention weights between query features \(\mathbf{Q}_\phi\) and landmark key features \(\mathbf{K}_{\text{land}}\) are computed via dot product (without scaling, since the features are already scaled by \(1/\sqrt{F}\) etc.):

\[
\mathbf{S}^{\text{land}} = \mathbf{Q}_\phi \mathbf{K}_{\text{land}}^\top \in \mathbb{R}^{B \times H \times N \times L}.
\]

The **softpick** activation (defined identically to that in LEKAv1 but now actually used) is applied to obtain sparse attention probabilities:

\[
\mathbf{A}^{\text{land}} = \text{Softpick}(\mathbf{S}^{\text{land}}, \text{dim}=-1).
\]

Softpick is a rectified softmax that can produce exact zeros, encouraging sparsity. A dropout layer is applied after softpick. The landmark context is then:

\[
\mathbf{C}_{\text{land}} = \mathbf{A}^{\text{land}} \tilde{\mathbf{V}}_{\text{land}} \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

### 2.6 Local Window Attention

In parallel, a simple sliding window attention is computed on the original (non‑feature) Q, K, V. A distance mask allows each token to attend within a small window of size \(W\) (local_window_size, default 5):

\[
\mathbf{S}^{\text{local}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}, \qquad
S^{\text{local}}_{ij} = -\infty \text{ if } |i-j| > \lfloor W/2 \rfloor.
\]

Softmax (not softpick) is applied, and the local context is:

\[
\mathbf{C}_{\text{local}} = \mathbf{A}^{\text{local}} \mathbf{V}.
\]

### 2.7 Per‑Head Gated Fusion

A per‑head learnable gate \(\gamma_h\) (initialised to \(0.1\)) determines the blending of the two branches:

\[
\mathbf{C} = \sigma(\gamma_h) \cdot \mathbf{C}_{\text{land}} + (1 - \sigma(\gamma_h)) \cdot \mathbf{C}_{\text{local}},
\]

broadcast over batch and sequence. This allows heads to specialise: some may rely more on global landmarks, others on local window.

### 2.8 Output Projection and Masking

The combined context is reshaped to \((B, N, D)\) and projected:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

Attention mask handling: The code accepts an `attention_mask` and converts various shapes (2D, 3D, 4D) to a 2D boolean `token_valid`. It masks the landmark attention scores by setting invalid positions to \(-10^9\) before softpick. However, it does **not** mask the local attention or the landmark key/value aggregation, which could allow padding tokens to affect the output; masking should be applied more thoroughly.

### 2.9 Additional Parameters

- `landmark_logits`: a learnable parameter (shape \(H\)) initialised to \(\log(L/2)\), likely intended to adjust the number of landmarks dynamically, but it is **not used** in the forward pass. Possibly a leftover.
- `gate`: a global scalar gate (not used in the current path) for a future global/local trade‑off; it is only created when `use_gate` is True and `bert_mode` is False, but not referenced in forward.

---

## 3. Implementation Details

### 3.1 Module Structure

`LEKAv2Attention` inherits from `nn.Module`. Key attributes:

- `query`, `key`, `value`, `out_proj`: linear layers (copied from pretrained).
- `rand_proj1`, `rand_proj2`: learnable random projection matrices.
- `landmark_scorer`: MLP to score key features for landmark selection.
- `W_corr`: per‑head correction matrix (initialised to identity).
- `local_gate`: per‑head gate parameter.
- `local_window_size`: window size for local attention.
- `dropout`: dropout layer applied to landmark attention weights.
- `softpick` static method.

### 3.2 Forward Pass Summary

1. Project hidden states to Q, K, V (multi‑head).
2. Compute \(\phi(Q), \phi(K)\) using dual‑scale projections.
3. Select landmarks: score K features, gather top‑L, apply correction matrix.
4. Compute landmark attention scores (dot product of Q features and landmark K features), mask with `token_valid`, apply softpick and dropout, multiply by corrected landmark values.
5. Compute local attention: scores via scaled dot product, mask window, softmax, multiply by V.
6. Blend with learned per‑head gate.
7. Reshape, project, return (output, None).

### 3.3 Injection

The injection function `inject_leka_v2` copies weights from the pretrained attention (`q_lin`, `k_lin`, etc.) to the corresponding linear layers, ensuring zero‑shot transfer. The other parameters (random projections, scorer, correction matrix, gates) are randomly initialised and trained.

---

## 4. Usage and Configuration

Activated by `"leka_v2"` in compression config. Relevant parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_landmarks` | 16 | Number of landmark tokens \(L\). |
| `feature_dim` | 128 | Dimension \(F\) (total feature dim = \(3F\)). |
| `local_window_size` | 5 | Width of local window. |
| `dropout` | 0.1 | Dropout rate on landmark attention weights. |
| `use_gate` | False | Whether to create an extra global gate (unused currently). |
| `gate_bias` | 0.0 | Bias for that gate (unused). |

---

## 5. Discussion

LEKAv2 presents a sophisticated balance between global and local attention. The landmarks compress the sequence into a constant‑sized set, enabling linear time global attention with a rich feature map. The correction matrix adds flexibility to adjust landmark values, helping to mitigate information loss from compression. The local window ensures that fine‑grained interactions are not missed. The softpick activation on landmarks encourages sparsity, which may improve interpretability and efficiency.

However, several aspects warrant attention. The landmark selection is non‑differentiable due to top‑k; gradients will not flow through the selection step, although the scorer can be trained using a straight‑through estimator or by using a soft top‑k. The code does not explicitly handle this; the top‑k operation will block gradients to the scorer. The `landmark_logits` parameter is unused, indicating incomplete development. The local attention does not incorporate the attention mask, so padding tokens will be included in the window unless the input is pre‑packed. Additionally, the correction matrix \(W_{\text{corr}}\) is applied to the values after gathering, which is a linear transformation shared across sequence positions; it may be beneficial to make it input‑dependent.

Despite these limitations, LEKAv2 demonstrates a comprehensive approach to efficient attention, combining kernel features, landmark compression, correction, sparse activation, and local gating. It serves as a strong baseline for future research in hybrid efficient Transformers.

---

## 6. Conclusion

We have presented LEKAv2, a landmark‑based kernel attention module with dual‑scale random features, a correction matrix, softpick sparsity, and a gated local window. Its design achieves linear complexity while retaining global context through compressed landmarks and local precision through a window. Integrated into the DistilBERT compression framework with zero‑shot weight transfer, LEKAv2 offers a versatile and efficient replacement for standard self‑attention. Future work will address the differentiability of landmark selection and extend masking support for all branches.
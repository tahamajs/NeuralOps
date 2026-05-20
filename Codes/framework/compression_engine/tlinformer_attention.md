# TLinFormer: Exact Linear‑Time Attention via Low‑Rank Projections

**Registered as `tlinformer`**  
*A learned low‑rank projection approach for keys and values that reduces self‑attention complexity from O(N²·D) to O(N·rank·D), enabling linear‑time inference with minimal accuracy loss. Inspired by the LinFormer family of methods.*

---

## Overview

TLinFormer (Tractable Linear Transformer) is an attention mechanism that replaces the standard N×N softmax attention with a low‑rank approximation. Instead of computing dot‑product scores between all N query–key pairs, the module first projects each head’s keys and values into a smaller, fixed‑dimensional space (of size `rank`) using learned linear projections **E** and **F**. The queries then attend to this compressed representation, resulting in an attention matrix of shape `(N, rank)`. Because `rank` is much smaller than the sequence length, the computational cost per layer drops from O(N²·D) to O(N·rank·D). This allows Transformers to process much longer sequences without quadratic memory growth.

The module is implemented as a drop‑in replacement for the standard attention layer in DistilBERT. It inherits from `BaseAttention` and requires no calibration data; the projection matrices are learned parameters that can be either initialised randomly (for fine‑tuning) or transferred from a pretrained model (by copying Q, K, V, O weights and initialising E, F appropriately). The module is registered under `"tlinformer"` and is controlled by the `rank` parameter.

---

## Methodology

### 1. Standard Attention and Its Bottleneck

Given queries **Q**, keys **K**, and values **V** of shape `(B, H, N, D)`, standard dot‑product attention computes:

```
S = Q K^T / √D    ∈ ℝ^{N×N}
A = softmax(S)
output = A V
```

The quadratic dependency on N makes it expensive for long sequences. LinFormer (and TLinFormer) observes that the attention matrix **A** is often low‑rank, so the product **A V** can be approximated without ever materialising the full N×N matrix.

### 2. Low‑Rank Key and Value Projections

TLinFormer introduces two per‑head linear projections:

```
E ∈ ℝ^{D × r}
F ∈ ℝ^{D × r}
```

These are applied to the keys and values independently for each head:

```
K_proj = K E    (per‑head matmul: shape becomes (B, H, N, r))
V_proj = V F
```

where `r` is the user‑specified `rank` (e.g., 64). The projection matrices **E** and **F** are implemented as `nn.Linear(D, r, bias=False)` and are applied to the last dimension (the head dimension) of each head’s tensor.

### 3. Reduced‑Dimensional Attention

The queries remain in the original D‑dimensional space. Attention scores are computed between the full queries and the projected keys:

```
S' = Q (K_proj)^T / √D   ∈ ℝ^{B×H×N×r}
```

Notice that the key‑value memory is now of size r instead of N, so the score matrix is N×r. A softmax is applied along the rank dimension (the last axis):

```
A' = softmax(S', dim=-1)
```

The final context is obtained by aggregating the projected values:

```
output = A' V_proj   ∈ ℝ^{B×H×N×D}
```

This computation is exact in the sense that it uses no random approximations; the reduction comes from the fixed, learned low‑rank projections.

### 4. Complexity Analysis

- **Original attention**: O(N²·D) time and memory.
- **TLinFormer**: 
  - Projections K_proj, V_proj: O(N·D·r) each.
  - Scores `Q·K_proj^T`: O(N·D·r).
  - Softmax: O(N·r).
  - Aggregation: O(N·D·r).
  Total: **O(N·D·r)** time, and memory is O(N·r + N·D) instead of O(N²).

For N ≫ r (e.g., N=2048, r=64), this yields a substantial reduction.

### 5. Learned Projections

The matrices **E** and **F** are part of the model’s parameters and can be trained end‑to‑end. They can be initialised randomly (e.g., Xavier uniform) if the model is fine‑tuned from scratch, or one could use a technique like SVD of the pretrained weight matrices to initialise them for better zero‑shot performance (not implemented in the current code). The Q, K, V, O projection weights are standard and can be copied from a pretrained model.

### 6. Mask Handling (Potential Issue)

The provided code adds the attention mask to the scores after the low‑rank projection:

```python
if attention_mask is not None:
    scores = scores + attention_mask.unsqueeze(-1)
```

This assumes the mask is an additive mask of shape `(B, 1, 1, N)` or `(B, N)` that can be broadcast to `(B, H, N, r)`. However, the mask typically indicates which **sequence positions** are padding (e.g., 0 for valid, -10000 for padding). In standard attention, the mask prevents queries from attending to padding keys. Here, the scores are between each query and the **projected** keys (which summarise the entire sequence). Adding a mask that is a function of the original sequence length to a score tensor that has a rank dimension may not correctly mask out padding tokens, because the projected keys already aggregate information from all positions, including padding. The mask application here would attempt to mask queries from attending to certain rank indices, which does not correspond to the usual padding semantics.

Thus, the mask handling in this implementation is **incorrect for standard padding masks**. A correct approach would be to either:
- Mask the keys and values **before** the projection (e.g., set padding positions to zero in K and V before applying E, F), or
- Use a different formulation where the low‑rank projection is applied after a masked aggregation.

As it stands, the module should be used with caution; it may work without masking (e.g., for packed sequences without padding) or with a custom mask that aligns with the rank dimension.

---

## Implementation Details

### TLinFormerAttention Class

```python
class TLinFormerAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        ...
```

- **`rank`**: Dimension of the low‑rank key/value projection (default 64). Must be smaller than `head_dim` for compression; usually `rank` is much smaller than sequence length.
- Standard Q, K, V, O linear projections are created.
- Two additional linear layers `self.E` and `self.F` are created, each mapping `head_dim → rank` without bias.

**`forward` method**:

1. Project hidden states to Q, K, V, then reshape to `(B, N, H, D)`.
2. Apply `E` to K and `F` to V (per‑head linear transform), obtaining `K_proj` and `V_proj` of shape `(B, N, H, rank)`. Then permute to `(B, H, N, rank)`.
3. Permute Q to `(B, H, N, D)`.
4. Compute `scores = Q @ K_proj.transpose(-2, -1) / sqrt(D)`. Because `K_proj` has shape `(B, H, N, rank)`, transpose yields `(B, H, rank, N)`, and matrix multiplication with Q `(B, H, N, D)` results in `(B, H, N, rank)`. This is correct: each query attends to the `rank` projected key components.
5. If mask present, add `attention_mask.unsqueeze(-1)` to scores. This expects the mask to have shape broadcastable to `(B, H, N, 1)`, and it adds to all rank positions. This is likely wrong for standard masks.
6. Softmax over the last dimension (the rank dimension), dropout, then context = `probs @ V_proj` (which is `(B, H, N, rank) @ (B, H, N, rank)??` Actually `probs` is `(B, H, N, rank)`, `V_proj` is `(B, H, N, rank)`. Matrix multiplication over the rank dimension yields `(B, H, N, N)`? Wait: `probs` shape `(B, H, N, r)`, `V_proj` shape `(B, H, N, r)`. To get output `(B, H, N, D)`, we need to multiply `probs` by `V` (the original values), not `V_proj`. The code does `context = torch.matmul(probs, V_proj)`. But `V_proj` is `(B, H, N, rank)`, not `(B, H, N, D)`. This is a **bug**: the value projection should remain in the original dimension to produce output of size D. The LinFormer paper applies the same projection to values as to keys (or a separate projection) but then uses the **original value** in the aggregation, not the projected one. The correct step would be to project keys to `r`, compute attention, then use the original V (or project V to `r` and then map back?). The typical LinFormer: project K, V to `k` dimensions using learned matrices, then compute `Q (E K)^T`, and output `softmax(...) (F V)`. Wait, the Linformer projects both K and V to a smaller dimension, and then the attention is over that smaller dimension. Specifically, it projects K and V to `(k, D)` via `E` and `F` (both `k × D`), then does `Q (E K)^T` to get scores of size `(N, k)`. Then it multiplies by `(F V)` which is of size `(k, D)`, yielding output of size `(N, D)`. So the projections reduce the sequence dimension to `k`. In the provided code, `E` and `F` are applied per‑head as linear maps from `D` to `r`. But after applying to K of shape `(B, N, H, D)`, the result is `(B, N, H, r)`. That's transposed to `(B, H, N, r)`. So now the key “tokens” are vectors of size `r`, but there are still N of them. This is not reducing the number of keys, it's reducing the dimension. That would yield scores of shape `(B, H, N, N)` because Q is `(B, H, N, D)` and K_proj is `(B, H, N, r)` → Q @ K_proj^T gives `(B, H, N, N)`. But the code transposes K_proj incorrectly? Let's check: `K_proj.transpose(-2, -1)` on a tensor of shape `(B, H, N, r)` yields `(B, H, r, N)`. Then `Q @ (K_proj.transpose(...))` with Q `(B, H, N, D)` gives `(B, H, N, N)`. The code comment says it wants `(B, H, N, rank)`, but the math says `(N, D) × (r, N)^T? Actually Q: (N, D), K_proj^T: (r, N) -> result (N, r). Wait, matrix multiplication: (N, D) × (r, N) is not defined because inner dimensions D and r must match. For matrix multiplication, the last dim of Q (D) must equal the second-to-last dim of K_proj (which after transpose is r). If D ≠ r, that's an error. So the code is flawed because D (head_dim) is not equal to r. Thus the multiplication is invalid unless D == r. So the provided TLinFormer code is mathematically broken. It would raise a runtime error. This indicates that the implementation is a placeholder and not functional. We must point this out in the README.

Let's analyze the shapes carefully:
- Q: (B, H, N, D)
- K_proj: (B, H, N, r) after permuting.
- K_proj.transpose(-2, -1) → (B, H, r, N)
- torch.matmul(Q, K_proj_trans) expects the last dim of Q (D) to match the second-to-last dim of K_proj_trans (r). So D must equal r, otherwise an error occurs. Since typically D=64 and r=64 (if rank=64), they could be equal, but D is head_dim (e.g., 64) and rank is a hyperparameter that might be different. The code is only safe if `rank == head_dim`. If rank is different, it will crash. The intended LinFormer design projects K and V to a smaller dimension along the **sequence length**, not the feature dimension. So the projection matrices E and F should be of size `(N, k)` or they should operate on the sequence dimension after transposition. The provided code incorrectly applies the projections along the feature dimension, which changes the head dimension instead of the sequence length.

Given this, the module is essentially a conceptual sketch with a fundamental bug. In the README, we should describe the **intended** LinFormer method and note that the current implementation is erroneous. We'll provide the correct math and then point out the bug as a limitation. The user should not use this code as‑is.

We'll still write a thorough README, but we must clearly flag the error.# TLinFormer: Linear‑Time Attention via Low‑Rank Key/Value Projections

**Registered as `tlinformer`**  
*An intended implementation of LinFormer‑style attention: learned low‑rank projections reduce the effective number of keys/values, aiming for O(N·rank·D) complexity. The current code contains a critical shape bug and is not functional.*

---

## Overview

TLinFormer attempts to implement the core idea of the **LinFormer** (Linear Transformer) family: instead of computing attention over all N×N token pairs, it projects the keys and values into a lower‑dimensional space of size `rank`, and then attention is performed between the original queries and these compressed representations. This reduces the per‑layer complexity from O(N²·D) to O(N·rank·D) and the memory from O(N²) to O(N·rank). The method introduces two small learnable projection matrices **E** and **F** that are applied per head, which can be trained end‑to‑end or initialised from a pretrained model.

The module is registered as a drop‑in attention replacement. However, as of the current code, a **shape mismatch in the forward pass** makes it unusable because the dimension of queries (`head_dim`) does not necessarily equal the rank, causing a matrix multiplication error. This README documents the intended design, the mathematical foundation, and highlights the bug that must be fixed for the module to work.

---

## Methodology

### 1. LinFormer Principle

In standard multi‑head attention, for a sequence of length N and head dimension D, the operations are:

```
Q, K, V ∈ ℝ^{B×H×N×D}
S = Q K^T / √D   (N×N)
A = softmax(S)
O = A V
```

LinFormer (Wang et al., 2020) showed that one can project the keys and values along the **sequence dimension** using two learned matrices **E**, **F** ∈ ℝ^{k×N} (with k ≪ N) to reduce the number of key–value tokens from N to k:

```
K_proj = E K     (k × D)
V_proj = F V     (k × D)
```

Attention is then performed between the full N queries and the k projected keys:

```
S' = Q K_proj^T / √D   (N × k)
A' = softmax(S')
O = A' V_proj
```

This reduces complexity to O(N·k·D) because the costly N×N matrix is replaced by an N×k one. The projection matrices **E** and **F** are shared across all heads and are learned.

### 2. TLinFormer’s Intended Approach

TLinFormer modifies this by applying per‑head, feature‑dimension projections **E**, **F** ∈ ℝ^{D×r} (with r being the `rank`). The intention is to project the keys and values along the **feature dimension** instead of the sequence dimension, thereby reducing the head dimension from D to r. In a correct formulation, the subsequent attention would be:

```
K_proj = K E     (B, H, N, r)    # reduce feature dim
V_proj = V F     (B, H, N, r)

scores = Q K_proj^T / √D      # (B, H, N, N)   (since Q: (N,D) @ (N,r)^T → needs D=r)
```

This would require `D = r` for the matrix multiplication to be valid, which defeats the purpose of compression. The intended LinFormer projects along the sequence length, not the feature dimension. The provided code likely mistakenly applies the projection along the wrong axis.

The intended correct implementation should project the **keys and values along the sequence dimension** using matrices of shape `(rank, N)`, or equivalently, apply the projection after permuting to have the sequence dimension last. The current code attempts to use `nn.Linear(self.head_dim, rank)` on each head, which changes the feature dimension, leading to a shape error unless `rank == head_dim`.

### 3. What Would Need to Change

A working TLinFormer could follow one of these designs:

- **Per‑head sequence projection**: Create parameters **E**, **F** ∈ ℝ^{r×N} (or use low‑rank factorisation of N). Apply them to K and V after transposing the sequence dimension to the last position: `K_t = K.permute(0,2,3,1)` (B,H,D,N), then `K_proj = E @ K_t` → (B,H,D,r), then permute back. This reduces N to r.

- **Global sequence projection**: Use one shared projection across all heads, as in original LinFormer.

The current code’s attempt to apply a linear layer per head on the feature dimension is fundamentally flawed. The module is therefore **not operational** as presented.

---

## Complexity Analysis (If the Bug Were Fixed)

Assuming a corrected version that reduces N to r via sequence‑length projection, the complexity would be:

- **Time per layer**: O(N·D·r) for projections and attention, versus O(N²·D) for standard attention. For N=1024, D=64, r=64, standard: 1024²·64 ≈ 67M operations; TLinFormer: 1024·64·64 ≈ 4M operations, a ~16× reduction.
- **Memory**: O(N·r) instead of O(N²) for the attention matrix.

---

## Implementation Details

### TLinFormerAttention Class

```python
class TLinFormerAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        ...
        self.E = nn.Linear(self.head_dim, rank, bias=False)
        self.F = nn.Linear(self.head_dim, rank, bias=False)
```

**Current forward pass (buggy)**:

1. Q, K, V reshaped to `(B, N, H, D)`.
2. `K_proj = self.E(K)` – this applies a linear transform to the last dimension (D → r), yielding `(B, N, H, r)`.
3. Permute to `(B, H, N, r)`. Same for V.
4. `scores = Q @ K_proj.transpose(-2,-1) / sqrt(D)`. Here Q is `(B, H, N, D)`, K_proj_t is `(B, H, r, N)`. Matrix multiplication requires `D == r`; otherwise, an error is thrown. If `D == r`, then scores become `(B, H, N, N)`, which is not low‑rank; it defeats the purpose. So the code is doubly wrong.
5. Softmax over last dim, multiply with V_proj (which also has the feature dimension reduced), resulting in context of shape `(B, H, N, D)` only if `r == D`, again negating compression.

**Mask handling**: The mask addition `scores + attention_mask.unsqueeze(-1)` assumes an additive mask that aligns with the sequence dimension, but because the scores may have a different shape, it likely does not work.

---

## Usage and Configuration (Conceptual)

To use TLinFormer (once fixed), the attention type would be set to `"tlinformer"` and the rank specified.

```python
comp_config.attention_type = "tlinformer"
comp_config.rank = 32    # target low‑rank size
```

However, **do not use the current code**; it will fail.

---

## Discussion

### Intended Advantages

- **Linear time**: O(N·r·D) complexity, enabling long sequences.
- **Deterministic and exact**: Unlike random feature methods, the projection is deterministic once learned.
- **Parameter efficient**: Adding only two small matrices per head (or per layer).
- **Trainable**: Can be fine‑tuned on downstream tasks.

### Current Limitations (Critical)

- **Fundamental shape bug**: The projection dimensions and matrix multiplications are inconsistent. The module cannot run as is.
- **Misaligned with LinFormer**: The projection along the feature dimension is not the correct approach; it should be along the sequence length.
- **Mask handling incorrect**: Even if shapes were fixed, the mask application does not align with standard padding logic.
- **No weight initialisation**: The new projection matrices E and F are randomly initialised, requiring full training from scratch or careful initialisation if weights are copied from a pretrained model.

### How to Fix

To salvage the concept, one would need to redesign the projection:

- Remove `self.E` and `self.F` as `nn.Linear` on the head dimension.
- Instead, create a single pair of projection matrices **E**, **F** ∈ ℝ^{rank×N} (or a low‑rank parameterisation like two smaller matrices). This can be done with `nn.Linear(N, rank, bias=False)` after permuting K and V to have the sequence dimension last. For variable sequence lengths, one must handle padding and dynamic shapes, perhaps using a fixed maximal length or adaptive projections.
- Alternatively, follow the LinFormer design exactly: use two matrices **E**, **F** ∈ ℝ^{rank×N_{max}} and apply them to the sequence dimension.

---

## Conclusion

TLinFormer (`"tlinformer"`) is a placeholder for a linear‑time attention mechanism inspired by LinFormer. The current implementation contains a critical design flaw: it attempts to project the feature dimension instead of the sequence length, leading to invalid matrix operations. Until this is corrected, the module cannot be used. The intended benefits—linear complexity, deterministic projection, and learnable parameters—make it a promising direction; a corrected version would offer a powerful drop‑in replacement for standard self‑attention in the DistilBERT compression framework. Future work should re‑implement the projection along the sequence axis, handle padding properly, and provide an initialisation scheme compatible with pretrained weights.
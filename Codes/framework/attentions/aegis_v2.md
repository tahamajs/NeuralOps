# Aegis‑Attention V2

Low‑rank projections + local dense window + global linear kernel attention + adaptive soft masking. A hybrid attention that captures both local precision and global context efficiently.

## What it does
Aegis‑Attention V2 replaces the standard quadratic self‑attention with a mixture of two attention mechanisms:

- **Local window attention** – dense softmax attention restricted to a fixed‑size sliding window (`local_window`). Captures fine‑grained local syntax and interactions.
- **Global linear attention** – kernelised linear attention using the feature map φ(x) = ELU(x) + 1. Computes long‑range interactions in O(N·d²) instead of O(N²).
- **Adaptive pruning gate** (optional) – a head‑specific sigmoid gate that down‑weights tokens before the global aggregation, effectively dropping irrelevant ones.
- **Adaptive fusion gate** – a learned scalar α per head and token that blends local and global outputs:  
  `output = α * local_out + (1‑α) * global_out`.
- **Low‑rank projections** – every Q, K, V, O projection is factorised into two matrices (`hidden → rank` and `rank → hidden`), initialised from the pretrained weights via truncated SVD, keeping total parameters low.

## Method
1. **Low‑rank SVD initialisation**  
   The original weight matrix \(W\) is decomposed as \(W \approx U_r S_r V_r^T\) using truncated SVD.  
   The two‑layer projection is initialised as:  
   `lin1.weight = V_r`, `lin2.weight = U_r @ diag(S_r)`.  
   Bias (if any) is copied into `lin2.bias`. This yields a strong zero‑shot approximation of the pretrained transformation.

2. **Local attention**  
   Q, K, V are reshaped into `(B, N, heads, head_dim)`.  
   Attention scores are computed as `Q·K^T / √d` and then masked to keep only the diagonal band of width `local_window`. Padding positions (from the attention mask) are also set to `-∞`. Softmax produces a sparse attention matrix. The output is `local_out`.

3. **Global linear attention**  
   Uses the kernel trick: φ(x) = ELU(x) + 1.  
   If adaptive pruning is enabled, K and V are multiplied by a head‑wise sigmoid gate `keep`, which can suppress tokens.  
   Then a key‑value memory matrix `KV = φ(K)^T @ V` and a normalisation vector `Z = φ(K)^T @ 1` are built in \(O(N d^2)\) time.  
   The query is projected through φ and then multiplied:  
   `global_out = (φ(Q) @ KV) / (φ(Q) @ Z + ε)`.  
   This yields a global summary that mixes all tokens in linear time.

4. **Fusion**  
   A gate `α = sigmoid(W_f * hidden_state)` (per head, per token) interpolates between the local and global outputs. This allows the model to dynamically decide which source of information is more relevant at each position.

5. **Output projection**  
   The fused representation is passed through another low‑rank `out_proj1 → out_proj2` pair, dropout, and finally masked to zero‑out padding tokens.

## Implementation
- **Config class**: `AegisAttentionConfigV2` holds all hyperparameters.
- **Module class**: `AegisAttentionV2` contains:
  - Three pairs of low‑rank projectors: `q_proj1/q_proj2`, `k_proj1/k_proj2`, `v_proj1/v_proj2`.
  - One output projector pair: `out_proj1/out_proj2`.
  - `prune_gate` – a small MLP (Linear→ReLU→Linear) that outputs a per‑head score for each token.
  - `fusion_gate` – a single linear layer outputting per‑head fusion coefficients.
- **Static method `_svd_init`** initialises the low‑rank pairs from a pretrained linear layer.
- The forward pass handles input masking flexibly: 2D, 3D, or 4D attention masks, converting them to a 2D boolean `(B, N)` where `True` means keep.

## How to use
In your compression config, set:
```python
comp_config.block_size = 64          # local window width
comp_config.lora_r = 64              # rank of low‑rank projections
comp_config.rff_features = 128       # (unused, kept for compatibility)
comp_config.dropout = 0.1
comp_config.use_gate = True          # enable adaptive pruning
```
The injection function registered as `"aegis_v2"` automatically replaces the layer’s attention.  
Manual injection:
```python
cfg = AegisAttentionConfigV2(
    hidden_size=model_config.hidden_size,
    num_attention_heads=...,
    head_dim=...,
    local_window=64,
    rank=64,
    kernel_feature_dim=128,
    dropout=0.1,
    use_adaptive_pruning=True
)
attn = AegisAttentionV2(cfg)
# run _svd_init for each projection using original layer
layer.attention = attn
```

## Configuration parameters
| Parameter          | Config key     | Default | Description |
|--------------------|----------------|---------|-------------|
| `local_window`     | `block_size`   | 64      | Width of the local attention window |
| `rank`             | `lora_r`       | 64      | Rank of all low‑rank projection pairs |
| `kernel_feature_dim` | `rff_features` | 128   | (Unused; kept for interface consistency) |
| `dropout`          | `dropout`      | 0.1     | Dropout rate on attention weights and output |
| `use_adaptive_pruning` | `use_gate`  | True    | Whether to apply the token‑wise pruning gate to K/V in global branch |

## Important notes
- **Computational complexity**: Local branch costs \(O(N \cdot W \cdot d)\) with window size \(W\). Global branch costs \(O(N \cdot d^2)\). For long sequences, the global branch dominates but remains linear in \(N\), making it feasible for very large contexts.
- **SVD initialisation** guarantees the projection starts from a near‑exact low‑rank replica of the pretrained weights. Only the factorised matrices and gates are trainable; biases (if present) are also updated.
- **Pruning gate** acts as a learnable sparsity mask, potentially reducing noise in the global memory. The gate is head‑wise, so different heads can prune different tokens.
- **Fusion α** is predicted from the full hidden state before attention, giving the model the ability to choose per‑token whether to rely on local or global context.
- **Mask handling**: Valid tokens are zeroed out in Q, K, V *before* attention, and the output is masked again, ensuring padding is never used.
- **Output**: The module always returns `(output, None)`, ignoring `output_attentions`. No attention weights are returned for memory efficiency.

---

*Local + global, dense + linear, all packed into one parameter‑efficient, SVD‑initialised module.*
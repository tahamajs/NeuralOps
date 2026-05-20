# Aegis‑Hybrid Attention

Low‑rank projections + local window attention + global linear attention + adaptive pruning. A hybrid that captures both local patterns and long‑range dependencies efficiently.

## What it does
Aegis‑Hybrid replaces standard quadratic self‑attention with a mixture of two attention mechanisms:

- **Local window attention** – standard softmax attention restricted to a sliding window of size `local_window`. Good for short‑range syntax.
- **Global linear attention** – kernelised (elu+1) linear attention that covers the whole sequence in linear time. Good for long‑range semantics.
- **Adaptive pruning** (optional) – a head‑wise gate that can drop tokens before the global linear pass, making it even sparser.
- **Adaptive fusion** – a learned gate that blends local and global outputs per token and per head.

All four projection matrices (Q, K, V, O) are themselves **low‑rank** (SVD‑initialised from the pretrained weights) to keep parameter counts small.

## Method
1. **Low‑rank projections** – each linear layer is split into two smaller matrices (`proj1` of shape `hidden→rank`, `proj2` of shape `rank→hidden`). They are initialised via truncated SVD of the original weight, giving a great zero‑shot starting point.
2. **Local window attention** – Q and K are multiplied, scaled, and then a band mask is applied so each token only attends to neighbours within `local_window`. Padding is also masked. Output: `local_out`.
3. **Global linear attention** – uses the feature map φ(x)=ELU(x)+1. The key‑value pairs are aggregated into a `(d,d)` matrix, and the query vector retrieves the global context in O(N) time.  
   Optionally, a **pruning gate** multiplies K and V by a sigmoid gating value before aggregation, allowing the model to drop irrelevant tokens.
4. **Fusion** – a gating scalar α is predicted from the hidden state (per head, per token) and used to interpolate: `output = α·local + (1‑α)·global`.
5. **Output projection** – another low‑rank linear layer.

## Implementation
The module is a single `nn.Module` that expects a `AegisHybridConfig` dataclass. It contains:
- Low‑rank projector pairs: `q_proj1/q_proj2`, `k_proj1/k_proj2`, `v_proj1/v_proj2`, `out_proj1/out_proj2`.
- An optional `prune_gate` (MLP with ReLU).
- A `fusion_gate` linear layer.
- A static method `_svd_init` that performs the SVD initialisation from a pretrained linear layer.

The injection function `inject_aegis_hybrid` creates the config, builds the module, calls `_svd_init` for each of the four projections using the original `old_attn.q_lin`, etc., and replaces the layer’s attention.

## How to use
Set your compression config values:
```python
comp_config.block_size = 64          # local window size
comp_config.lora_r = 64              # rank of low‑rank projections
comp_config.rff_features = 128       # kernel feature dimension (unused currently)
comp_config.dropout = 0.1
comp_config.use_gate = True          # enable adaptive pruning
```

The injection is registered under `"aegis_hybrid"` and happens automatically.

Manual injection for a layer:
```python
cfg = AegisHybridConfig(
    hidden_size=model_config.hidden_size,
    num_attention_heads=...,
    head_dim=...,
    local_window=64,
    rank=64,
    kernel_feature_dim=128,
    dropout=0.1,
    use_adaptive_pruning=True
)
attn = AegisHybridAttention(cfg)
# run _svd_init for each projection using the original attention module
layer.attention = attn
```

## Configuration parameters
| Parameter | Config key | Default | Description |
|-----------|------------|---------|-------------|
| `local_window` | `block_size` | 64 | Width of the local attention window |
| `rank` | `lora_r` | 64 | Rank of the low‑rank projection pairs |
| `kernel_feature_dim` | `rff_features` | 128 | (currently unused; kept for compatibility) |
| `dropout` | `dropout` | 0.1 | Dropout on attention weights and output |
| `use_adaptive_pruning` | `use_gate` | True | Whether to gate K,V before global linear attention |

## Important notes
- **Hybrid computation** – the local window part still costs O(N·W) with window size W; the global part is O(N·d²). For long sequences the global part dominates but stays linear.
- **SVD initialisation** – ensures the low‑rank projections start from a high‑fidelity approximation of the pre‑trained weights. Only the `proj1` and `proj2` matrices are trained (plus biases, gates).
- **Pruning gate** – if enabled, a token‑wise sigmoid gate reduces the contribution of unimportant tokens in the global aggregation, acting as a learnable sparsity mask.
- **Mask handling** – padding tokens are zeroed out in Q, K, V before attention and also masked in the local scores. Output is also zeroed for padding positions.
- **No attention weights returned** – `output_attentions` is ignored; the module always returns `(output, None)`.

---

*Local + global, low‑rank + adaptive. Good for efficient long‑sequence modelling.*
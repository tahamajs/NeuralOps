<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `ProgressiveTokenPruning` Class

```python
class ProgressiveTokenPruning(BaseAttention):
    def __init__(self, config=None, keep_ratio=0.8, rank=None, ...):
        super().__init__(config, rank)
        
        # ... resolve dims ...
        
        # Score network
        self.score_network = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Q/K/V/O with LoRA
        self.q_lin = LoRALinear(...)
        self.k_lin = LoRALinear(...)
        self.v_lin = LoRALinear(...)
        self.out_lin = LoRALinear(...)
```

## B.2 Scoring and Selection

```python
# Score tokens
scores = torch.sigmoid(self.score_network(x)).squeeze(-1)  # (B, L)

# CLS preservation
scores[:, 0] = 1e4

# Padding mask
valid_mask = get_token_valid_mask(attention_mask, B, L, device)
scores = scores.masked_fill(~valid_mask, -1e4)

# Re-apply CLS (padding mask may have overwritten)
scores[:, 0] = 1e4

# Top-k
k_tokens = max(1, min(L, int(math.ceil(L * self.keep_ratio))))
_, top_indices = torch.topk(scores, k_tokens, dim=-1)
top_indices, _ = torch.sort(top_indices, dim=-1)
```

## B.3 KV Gather

```python
# Repeat KV for GQA if needed
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    k_all = repeat_kv(k_all, repeats)
    v_all = repeat_kv(v_all, repeats)

# Gather K, V at top_indices
top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(
    -1, self.num_heads, -1, self.head_dim
)
k_pruned = torch.gather(k_all, 2, top_idx_exp)
v_pruned = torch.gather(v_all, 2, top_idx_exp)
```

## B.4 Causal Masking After Pruning

```python
q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
k_orig_pos = top_indices.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k)
causal_mask = k_orig_pos > q_pos
attn_scores = attn_scores.masked_fill(causal_mask, -1e4)
```

## B.5 Padding Mask for Pruned Keys

```python
valid_keys = torch.gather(valid_mask, 1, top_indices)  # (B, k)
attn_scores = attn_scores.masked_fill(
    ~valid_keys.unsqueeze(1).unsqueeze(2), -1e4
)
```

## B.6 Common Pitfalls

### B.6.1 Forgetting [CLS]

Without `scores[:, 0] = 1e4`, [CLS] may be pruned and classification fails.

### B.6.2 Padding Mask Overwrites [CLS]

If [CLS] position is not guaranteed valid, the padding mask overrides it. 
Re-apply the [CLS] clamp **after** the padding mask.

### B.6.3 Causal Mask Uses Pruned Positions

Use **original** positions (`top_indices`) for causality, not pruned 
positions.

### B.6.4 Wrong GQA Order

Repeat KV **before** pruning, not after. Pruning before repeat would prune 
fewer effective heads.

### B.6.5 No Sort After Top-K

Sorting `top_indices` preserves order for gather and downstream layers.

### B.6.6 Unsorted Indices Break Downstream

Without sorting, gather produces an unsorted sequence, which breaks 
downstream layers that assume ordering.

---
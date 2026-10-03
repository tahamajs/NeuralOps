<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Pruning Function

```python
def _fasa_prune(self, k, v, mask=None):
    B, H, L, D_h = k.shape
    if L <= 8:
        return k, v, mask, None
    
    if self.importance_method == "l2":
        importance = torch.norm(k.float(), p=2, dim=-1).mean(dim=1)  # (B, L)
    else:
        importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)
    
    if mask is not None:
        mask_2d = get_token_valid_mask(mask, B, L, k.device)
        importance = importance.masked_fill(~mask_2d, -10000.0)
    
    if self.protect_first_token:
        importance[:, 0] = 1e9
    
    keep = max(8, min(L, int(L * self.fasa_keep_ratio)))
    _, idx = torch.topk(importance, keep, dim=-1)
    idx, _ = torch.sort(idx, dim=-1)
    
    idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
    k_pruned = k.gather(2, idx_exp)
    v_pruned = v.gather(2, idx_exp)
    
    pruned_mask = None
    if mask is not None:
        mask_2d = get_token_valid_mask(mask, B, L, k.device)
        pruned_mask = mask_2d.gather(1, idx)
    
    return k_pruned, v_pruned, pruned_mask, idx
```

## B.2 The Forward Path

```python
is_causal = bool(self.is_decoder and not is_cross and L > 1)

if is_causal or is_cross:
    # Standard full attention — no pruning
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = (
            torch.triu(torch.full((L, L_kv), float("-inf"), ...), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores + causal_mask
    # ... apply mask, softmax, etc.
else:
    # FASA pruning
    k_p, v_p, p_mask, kept_idx = self._fasa_prune(k, v, eff_mask)
    attn_scores = torch.matmul(q.float(), k_p.float().transpose(-2, -1)) * scale
    if p_mask is not None:
        attn_scores = attn_scores.masked_fill((p_mask == 0).unsqueeze(1).unsqueeze(2), -10000.0)
    attn_weights = F.softmax(attn_scores, dim=-1).to(dtype)
    out = (
        torch.matmul(attn_weights, v_p)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(B, L, self.all_head_size)
    )
```

## B.3 Weight Transfer

Standard four-architecture handling with additional GQA pooling logic:

```python
if proj_key in ("k", "v"):
    if src_out == out_d:
        target_w.data.copy_(w_oriented)
    elif src_out > out_d and src_out % out_d == 0:
        rep = src_out // out_d
        target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
    elif src_out < out_d and out_d % src_out == 0:
        rep = out_d // src_out
        target_w.data.copy_(w_oriented.repeat(rep, 1))
    else:
        # Partial copy
        ...
```

**This handles GQA (Qwen)** where $H_{kv} < H$:
- If source K has more heads than target, mean-pool.
- If source K has fewer heads, repeat.

## B.4 Common Pitfalls

### B.4.1 Decoder Path Doesn't Prune

Users expecting speedup on GPT-2 will be disappointed. Document this.

### B.4.2 Pruning Every Forward

The token subset $\mathcal{S}$ is recomputed every forward. Not cached.

### B.4.3 Padding Tokens

Padding tokens are masked via `importance.masked_fill(~mask_2d, -10000.0)`. 
This ensures padding is not selected.

### B.4.4 Minimum Keep Count

`keep = max(8, min(L, int(L * fasa_keep_ratio)))` prevents over-pruning on 
short sequences.

### B.4.5 CLS Pinning Cannot Be Overridden

If `protect_first_token=False`, the CLS token may be pruned, causing 
accuracy collapse. This is a foot-gun.

## B.5 Diagnostics

### Count Kept Tokens

```python
with torch.no_grad():
    _, _, _, idx = model._fasa_prune(k, v)
    print(f"Kept: {idx.shape[-1]} / {k.shape[2]}")
```

### Visualize Kept Positions

```python
import matplotlib.pyplot as plt

kept = idx[0].cpu().numpy()
mask = np.zeros(128)
mask[kept] = 1
plt.imshow(mask[None, :], aspect="auto")
plt.title("Kept positions")
plt.show()
```

## B.6 Registry

```python
@ATTENTION_REGISTRY.register("fasa")
@ATTENTION_REGISTRY.register("fasa_attention")
@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention): ...
```

## B.7 Injection

Standard whole-model and single-layer injection with nested-encoder 
detection.

---
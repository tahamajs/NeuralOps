<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `softpick` Function

```python
def softpick(x, dim=-1, temperature=1.0):
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(m < -1e3, torch.zeros_like(m), m)  # clamp for masked rows
    z = x - m
    exp_z = torch.exp(z / temperature)
    exp_neg_m = torch.exp((-m) / temperature).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    zero_mask = denom <= 1e-8
    if zero_mask.any():
        fallback = F.softmax(x, dim=dim)
        p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
    else:
        p = p / denom
    return p.to(orig_dtype)
```

**Note.** The clamp `max=1e4` on `exp_neg_m` prevents overflow. The 
`zero_mask.any()` triggers fallback for the batch.

## B.2 Local Window Attention

```python
def _local_window_attention(self, Q, K, V, valid_mask, is_causal):
    B, H, L, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    
    idx = torch.arange(L, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = dist > self.window_size
    if is_causal:
        mask = mask | (idx.unsqueeze(0) > idx.unsqueeze(1))
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -10000.0)
    
    pad_k = (~valid_mask).unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(pad_k, -10000.0)
    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
    
    attn = F.softmax(scores, dim=-1).to(Q.dtype)
    attn = self.dropout(attn)
    return torch.matmul(attn, V)
```

## B.3 Sparse Block Attention (Python Loop)

```python
def _sparse_block_attention(self, Q, K, V, valid_mask, is_causal):
    # ... pad to multiple of block_size
    nb = Lp // self.block_size
    
    # Compute block-level scores
    Q_mean = Qb.mean(dim=3)
    K_mean = Kb.mean(dim=3)
    blk_scores = torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1)) * scale
    
    if is_causal:
        blk_scores = blk_scores + causal_blk_mask
    
    top_k = min(self.top_k, nb)
    _, top_idx = blk_scores.topk(top_k, dim=-1)
    if is_causal:
        for i in range(nb):
            top_idx[:, :, i, 0] = i
    else:
        top_idx[:, :, :, 0] = 0
    
    out_blocks = torch.zeros_like(Qb)
    for i in range(nb):  # ← PYTHON LOOP
        idx = top_idx[:, :, i, :]
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
        K_sel = torch.gather(Kb, 2, idx_exp).reshape(B, H, top_k * self.block_size, d)
        V_sel = torch.gather(Vb, 2, idx_exp).reshape(B, H, top_k * self.block_size, d)
        
        scores_i = torch.matmul(Qb[:, :, i].float(), K_sel.float().transpose(-2, -1)) * scale
        
        # Causal mask for selected tokens
        # ... 
        
        attn_i = softpick(scores_i, dim=-1).to(Q.dtype)
        out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)
    
    return out_blocks.view(B, H, Lp, d)[:, :, :L, :]
```

## B.4 Global Spectral RFF

```python
def _global_spectral_rff_attention(self, Q, K, V, valid_mask, is_causal):
    B, H, L, d = Q.shape
    pad = (~valid_mask).unsqueeze(1).unsqueeze(-1)
    
    Qf = torch.matmul(Q.float(), self.rff_proj.float())
    Kf = torch.matmul(K.float(), self.rff_proj.float())
    norm = 1.0 / math.sqrt(self.rff_features)
    phi_Q = torch.cat([torch.cos(Qf), torch.sin(Qf)], dim=-1) * norm
    phi_K = torch.cat([torch.cos(Kf), torch.sin(Kf)], dim=-1) * norm
    
    phi_K = phi_K.masked_fill(pad, 0.0)
    V_masked = V.float().masked_fill(pad, 0.0)
    
    if is_causal:
        KV = phi_K.unsqueeze(-1) * V_masked.unsqueeze(-2)  # [B, H, L, 2F, d]
        KV_cumsum = torch.cumsum(KV, dim=2)
        num = torch.matmul(phi_Q.unsqueeze(-2), KV_cumsum).squeeze(-2)
        Z = torch.cumsum(phi_K, dim=2)
        den = (phi_Q * Z).sum(dim=-1, keepdim=True) + 1e-5
        return (num / den).to(Q.dtype)
    else:
        KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
        Z = phi_K.sum(dim=-2, keepdim=True)
        num = torch.matmul(phi_Q, KV)
        den = torch.matmul(phi_Q, Z.transpose(-2, -1)) + 1e-5
        return (num / den).to(Q.dtype)
```

**The `KV` tensor is the memory bottleneck** for causal path.

## B.5 The `inject_nasc` Exception

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e  # ← Re-raises!
```

**Contrast with other modules** that use `return None`.

## B.6 Common Pitfalls

### B.6.1 Encoder Default

Without setting `enable_cascade_on_encoders=True`, encoders use dense.

### B.6.2 `softpick` Fallback

The fallback is batch-level. If any row has zero denom, the whole batch 
uses softmax for that row (via `torch.where`), but the trigger is 
batch-wide.

### B.6.3 RFF Cumsum

The `KV_cumsum` tensor can be large. Consider fused kernel for long 
sequences.

### B.6.4 Python Loop in Sparse Block

The `for i in range(nb)` loop is not vectorized. For small nb it's fine.

### B.6.5 `inject_nasc` Raises

Unlike `inject_cosformer` or `inject_fasa`, this raises. Handle with 
try/except in production.

## B.7 Diagnostics

### Check Path

```python
logger.setLevel(logging.DEBUG)
model = ...
# Should print "path=dense" or "path=cascade"
```

### Check Gate

```python
summary = x.mean(dim=1)
gates = F.softmax(model.gate_proj(summary), dim=-1)
print(f"Gate: {gates.mean(dim=0)}")
```

### Check RFF Memory

```python
def measure_rff_memory(B, H, L, F, d):
    return B * H * L * 2 * F * d * 4 / 1e6  # MB
print(measure_rff_memory(8, 12, 128, 64, 64))  # ~50 MB
```

---
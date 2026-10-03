<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 FactorisedLinear

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, ..., max_rank=64):
        self.rank = max(min_rank, min(max_rank, min(in_features, out_features)))
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=True)

    def decompose_from_weight(self, weight, bias=None):
        U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total.clamp(min=1e-8)
        rank = int((cum < self.energy).sum().item()) + 1
        rank = max(self.min_rank, min(rank, min(weight.shape), self.max_rank))
        
        A = U[:, :rank] * S[:rank].unsqueeze(0)
        B = Vt[:rank, :]
        self.down.weight.data = B
        self.up.weight.data = A
```

## B.2 Head Split

```python
self.num_full_heads = max(1, self.num_heads // 3)
self.num_kernel_heads = self.num_heads - self.num_full_heads
```

## B.3 Kernel Heads (Linear Attention)

```python
def _kernel_linear_attention(self, Q, K, V):
    proj = self.rff_proj[:Hk].float()
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q * scale, proj)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K * scale, proj)
    Q_feat = torch.cat([torch.cos(Q_proj), torch.sin(Q_proj)], dim=-1) / math.sqrt(2 * F_dim)
    K_feat = torch.cat([torch.cos(K_proj), torch.sin(K_proj)], dim=-1) / math.sqrt(2 * F_dim)
    
    if self.is_decoder:
        for c in range(num_chunks):
            # chunked causal attention
            ...
    else:
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
    return ctx
```

## B.4 Full Heads (Softpick)

```python
def _full_softpick_attention(self, Q, K, V, token_valid=None):
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if self.is_decoder:
        causal_mask = torch.tril(...)
        scores = scores.masked_fill(~causal_mask, -1e4)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid, -1e4)
    attn_w = softpick(scores, dim=-1)
    return torch.matmul(attn_w, V)
```

## B.5 The `copy_weights` Bug

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")  # ← BUG
```

**Fix:**

```python
except Exception as e:
    logger.debug(f"UniversalAdapter failed: {e}")
    # continue
```

## B.6 The `rff_proj` Bug

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=False)
```

**Fix:**

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=True)
```

## B.7 Softpick Masked Row Handling

If all keys are masked, `p.sum() = 0`, output is 0. **Fix:**

```python
def softpick(x, dim=-1):
    p = F.relu(torch.exp(x.float()) - 1.0)
    denom = p.sum(dim=dim, keepdim=True)
    fallback = F.softmax(x.float(), dim=dim)
    p = torch.where(denom > 1e-8, p / (denom + 1e-8), fallback)
    return p.to(x.dtype)
```

## B.8 Common Pitfalls

### B.8.1 `FactorisedLinear.weight` Materialization

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

Materializes the full matrix. Used in Frobenius check.

### B.8.2 `rff_proj` Shape

$(H_k, d_h, F)$. For $H_k = 8$, $d_h = 64$, $F = 256$: 131K floats.

### B.8.3 Python Loop in Decoder

For $N = 128$, chunk=64: 2 iterations. Acceptable.

### B.8.4 `num_full_heads` Minimum

```python
self.num_full_heads = max(1, self.num_heads // 3)
```

For $H = 2$: $H_f = 1$, $H_k = 1$.

## B.9 Registry

```python
@ATTENTION_REGISTRY.register("orion")
@ATTENTION_REGISTRY.register("orion_attention")
@ATTENTION_REGISTRY.register("orion_module")
class ORIONAttention(BaseAttention): ...
```

## B.10 Injection

Standard traversal with double-injection guard.

---
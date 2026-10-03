<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Orthogonal Matrix Construction

```python
def _build_orthogonal_matrix(self, d, m):
    blocks = []
    remaining = m
    while remaining > 0:
        block_size = min(d, remaining)
        g = torch.randn(d, block_size)
        q, _ = torch.linalg.qr(g)
        blocks.append(q)
        remaining -= block_size
    W = torch.cat(blocks, dim=1) * math.sqrt(d)
    return W.t().contiguous()
```

**Output shape:** $(F, d)$.

## B.2 Feature Map

```python
def _feature_map(self, x):
    proj = torch.matmul(x, self.omega.to(dtype=x.dtype).t())
    diag_sq = torch.sum(x**2, dim=-1, keepdim=True) * 0.5
    max_proj = proj.amax(dim=-1, keepdim=True).detach()
    diff = torch.clamp(proj - max_proj - diag_sq, min=-30.0, max=10.0)
    phi = torch.exp(diff) + self.eps
    phi = phi / (math.sqrt(self.num_features) + self.eps)
    return phi
```

**Stability:**
- `max_proj.detach()` — no gradient through max.
- `clamp(-30, 10)` — bounded.
- `+eps` — positive.

## B.3 Causal Chunked Scan

```python
if self.is_decoder:
    M = Q_feat.shape[-1]
    chunk_size = min(64, N)
    num_chunks = (N + chunk_size - 1) // chunk_size
    S = torch.zeros(B, H, M, d, dtype=torch.float32)
    z = torch.zeros(B, H, M, dtype=torch.float32)
    
    for c in range(num_chunks):
        # ... intra, inter, update S, z
```

For $N = 128$: 2 iterations.

## B.4 The `inject_performer` Raise

```python
try:
    new_attn = PerformerAttention(...)
except Exception as e:
    logger.warning("PerformerAttention construction/weight copying failed: %s", e)
    raise RuntimeError(f"Injection failed")  # ← BUG
```

**Fix:**

```python
except Exception as e:
    logger.warning("PerformerAttention construction failed: %s", e)
    return None
```

## B.5 The `output_scale` Parameter

```python
class LoRALinear(nn.Module):
    def __init__(self, ...):
        if not hasattr(self, 'output_scale'):
            self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

**Should be removed** — no theoretical justification.

## B.6 Cross-Attention Bypass

```python
if encoder_hidden_states is not None:
    return hidden_states, None
```

Returns input unchanged.

## B.7 Common Pitfalls

### B.7.1 Feature Map Overflows Without Stabilization

Without max-subtraction: `exp(ω^T x)` overflows for `||x|| > 5`.

### B.7.2 Injection Raises

Wrap in try/except if None-return behavior is needed.

### B.7.3 `output_scale` Missing Key

On `load_state_dict(strict=True)`, this parameter is missing.

### B.7.4 Cross-Attention Silent Bypass

No warning when cross-attention is requested.

## B.8 Diagnostics

### Verify Feature Positivity

```python
Q_feat = model._feature_map(Q)
assert (Q_feat >= 0).all()
```

### Verify Omega Persistence

```python
sd = model.state_dict()
assert any("omega" in k for k in sd)
```

### Verify Causal

```python
x = torch.randn(1, 128, 768, requires_grad=True)
y = model(x)
y[0, 50].sum().backward()
assert x.grad[0, 51:].abs().max() < 1e-6
```

## B.9 Registry

```python
@ATTENTION_REGISTRY.register("performer")
@ATTENTION_REGISTRY.register("performer_attention")
class PerformerAttention(BaseAttention): ...
```

---
# Appendix B. Implementation Details

## B.1 The Kernel Feature Map

```python
def kernel_feature_map(self, x):
    x_norm = self.norm(x)  # LayerNorm on head_dim
    features = torch.einsum("bhnd,hdf->bhnf", x_norm, self.proj.to(x.dtype))
    scale = 1.0 / math.sqrt(self.feature_dim)
    phi_cos = torch.cos(features) * scale
    phi_sin = torch.sin(features) * scale
    return torch.cat([phi_cos, phi_sin], dim=-1)  # (B, H, N, 2F)
```

## B.2 QR Projection Init

```python
proj_data = torch.randn(self.num_heads, self.head_dim, self.feature_dim)
for h in range(self.num_heads):
    if self.head_dim < self.feature_dim:
        q_h, _ = torch.linalg.qr(proj_data[h].T)
        proj_data[h] = q_h.T
    else:
        q_h, _ = torch.linalg.qr(proj_data[h])
        proj_data[h] = q_h[:, : self.feature_dim]
self.register_buffer("proj", proj_data, persistent=True)
```

## B.3 Causal Forward

```python
kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_features.float(), v.float())
KV_cum = torch.cumsum(kv_outer, dim=2)
num = torch.einsum("bhnf,bhnfd->bhnd", Q_features.float(), KV_cum)
z_cum = torch.cumsum(K_features.float(), dim=2)
denom = torch.einsum("bhnf,bhnf->bhn", Q_features.float(), z_cum).unsqueeze(-1).clamp(min=1e-5)
context = (num / denom).to(hidden_states.dtype)
```

## B.4 Bidirectional Forward

```python
KV = torch.einsum("bhnf,bhnd->bhfd", K_features.float(), v.float())
K_sum = K_features.float().sum(dim=2, keepdim=True)
num = torch.einsum("bhnf,bhfd->bhnd", Q_features.float(), KV)
denom = (
    torch.einsum("bhnf,bhf->bhn", Q_features.float(), K_sum.squeeze(2))
    .unsqueeze(-1)
    .clamp(min=1e-5)
)
context = (num / denom).to(hidden_states.dtype)
```

## B.5 The Preservation Gate

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))

# In forward:
alpha = torch.sigmoid(self.pres_gate)
out = alpha * out + (1.0 - alpha) * hidden_states
```

## B.6 Fatal Error Handling (BUG)

**Current:**
```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

**Recommendation:**
```python
except Exception as e:
    logger.warning(f"UniversalAdapter extraction failed: {e}")
    # continue with fallbacks
```

## B.7 Common Pitfalls

### B.7.1 Cumsum Memory

The causal cumsum materializes `(B, H, L, 2F, d_h)` — can OOM at 
large L.

### B.7.2 FP32 Precision

We use `.float()` for the einsum to avoid FP16 precision loss in the 
cumsum.

### B.7.3 Mask Application

Mask must be applied **before** aggregation:

```python
K_features = K_features * mask
v = v * mask
```

### B.7.4 Feature Map Output

$\phi(x)$ has shape $(B, H, N, 2F)$, not $(B, H, N, F)$.

## B.8 Diagnostic Checks

```python
# After init
assert self.proj.shape == (H, d_h, F)

# After forward
assert context.shape == (B, H, N, d_h)
assert not torch.isnan(context).any()
```

---
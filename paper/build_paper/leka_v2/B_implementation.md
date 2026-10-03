# Appendix B. Implementation Details

## B.1 Dual-Scale Feature Map

```python
def kernel_feature_map(self, x):
    W1 = self.rand_proj1.to(x.dtype)
    W2 = self.rand_proj2.to(x.dtype)
    
    proj1 = torch.matmul(x, W1)
    f1 = torch.cat([torch.cos(proj1), torch.sin(proj1)], dim=-1) / math.sqrt(self.feature_dim)
    
    proj2 = torch.matmul(x, W2)
    f2 = torch.cat([torch.cos(proj2), torch.sin(proj2)], dim=-1) / math.sqrt(self.feature_dim // 2)
    
    return torch.cat([f1, f2], dim=-1)
```

## B.2 Landmark Selection

```python
def select_landmarks(self, K_features, V, valid_mask=None, num_landmarks=None):
    B, H, N, feat_dim = K_features.shape
    target = num_landmarks or self.num_landmarks
    L_max = min(target, N)
    
    scores = self.landmark_scorer(K_features).squeeze(-1)  # (B, H, N)
    if valid_mask is not None:
        scores = scores.masked_fill((~valid_mask).unsqueeze(1), -1e4)
    
    # Force position 0
    scores[:, :, 0] = 1e4
    _, idx = scores.topk(L_max, dim=-1)
    idx = idx.clone()
    idx[:, :, 0] = 0
    
    K_land = torch.gather(K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim))
    V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
    return K_land, V_land, idx
```

## B.3 Softpick

```python
@staticmethod
def softpick(x, dim=-1, temperature=1.0):
    x = x.float() / temperature
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(m < -100.0, torch.zeros_like(m), m)
    z = x - m
    exp_z = torch.exp(z)
    exp_neg_m = torch.exp(-m).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    is_empty = denom < 1e-6
    p_safe = p / denom.clamp(min=1e-6)
    uniform = torch.ones_like(p) / p.shape[dim]
    return torch.where(is_empty.expand_as(p), uniform, p_safe).to(x.dtype)
```

## B.4 Local Attention

```python
idx = torch.arange(N)
dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
window = min(self.local_window_size, N)
window_mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)

local_scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
local_scores = local_scores.masked_fill(~window_mask, -1e4)

if self.is_decoder:
    causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
    local_scores = local_scores.masked_fill(~causal, -1e4)

local_attn = F.softmax(local_scores, dim=-1).to(v.dtype)
context_local = torch.matmul(local_attn, v)
```

## B.5 Fusion

```python
gate_val = torch.sigmoid(self.local_gate).view(1, H, 1, 1)
context = (1.0 - gate_val) * context_local + gate_val * context_leka
```

**Note the order.** With $w_h = -2$, $g_h \approx 0.12$, so $(1-g_h) \approx 0.88$ — local path dominates.

## B.6 Common Pitfalls

### B.6.1 Fatal Error Handling

The `copy_weights` still raises `RuntimeError` on extractor failure. This 
kills the module.

### B.6.2 Not Truly Linear

At default $\rho = 0.5$, the landmark attention is $O(L^2/2)$ — 
quadratic.

### B.6.3 Correction Matrix Scaling

For large $d_h$ (>128), the correction matrix is expensive.

## B.7 Diagnostic Checks

```python
# After init
assert self.rand_proj1.shape == (d_h, F)
assert self.rand_proj2.shape == (d_h, F // 2)
assert self.W_corr.shape == (H, d_h, d_h)

# After forward
assert context.shape == (B, H, N, d_h)
assert not torch.isnan(context).any()
```

---
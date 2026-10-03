<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Feature Map

```python
if self.act_fun == "softplus":
    Q = F.softplus(Q.float())
    K = F.softplus(K.float())
else:
    Q = F.elu(Q.float()) + 1.0
    K = F.elu(K.float()) + 1.0
V = V.float()
```

Both Q and K are mapped. V is not (values are unconstrained).

## B.2 Cos/Sin Reweighting

```python
def _get_cos_sin_weights(self, seq_len, device, dtype):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    m = max(self.max_seq_len, seq_len)
    angles = (math.pi / (2.0 * m)) * pos
    cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
    sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
    return cos_weight, sin_weight
```

The weights are recomputed per forward. For fixed $L$, this is wasteful 
but correct.

## B.3 Encoder Path

```python
KV_cos = torch.matmul(K_cos.transpose(-2, -1), V)
KV_sin = torch.matmul(K_sin.transpose(-2, -1), V)

Z_cos = K_cos.sum(dim=2, keepdim=True)
Z_sin = K_sin.sum(dim=2, keepdim=True)

num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)
denom = (
    torch.matmul(Q_cos, Z_cos.transpose(-2, -1))
    + torch.matmul(Q_sin, Z_sin.transpose(-2, -1))
    + self.eps
)

context = (num / denom.clamp(min=self.eps)).to(dtype)
```

**6 matmuls, 2 sums.** Clean.

## B.4 Decoder Path (Chunked)

```python
chunk_size = 32
if N <= chunk_size:
    # Direct computation with causal mask
    causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
    score_c = torch.matmul(Q_cos, K_cos.transpose(-1, -2)).masked_fill(~causal_mask, 0.0)
    score_s = torch.matmul(Q_sin, K_sin.transpose(-1, -2)).masked_fill(~causal_mask, 0.0)
    scores = score_c + score_s
    num = torch.matmul(scores, V)
    denom = score_c.sum(dim=-1, keepdim=True) + score_s.sum(dim=-1, keepdim=True) + self.eps
    context = (num / denom.clamp(min=self.eps)).to(dtype)
else:
    # Chunked scan
    pad_len = (chunk_size - (N % chunk_size)) % chunk_size
    # ... pad and reshape
    for c in range(num_chunks):
        num_inter = torch.matmul(qc_c[:, :, c], S_cos) + torch.matmul(qs_c[:, :, c], S_sin)
        denom_inter = torch.matmul(qc_c[:, :, c], Z_cos) + torch.matmul(qs_c[:, :, c], Z_sin)
        num_list.append(num_intra[:, :, c] + num_inter)
        denom_list.append(denom_intra[:, :, c] + denom_inter)
        # Update state
        S_cos = S_cos + torch.matmul(kc_curr.transpose(-1, -2), v_curr)
        S_sin = S_sin + torch.matmul(ks_curr.transpose(-1, -2), v_curr)
        Z_cos = Z_cos + kc_curr.sum(dim=-2, keepdim=True).transpose(-1, -2)
        Z_sin = Z_sin + ks_curr.sum(dim=-2, keepdim=True).transpose(-1, -2)
    # Concatenate
```

**4 outer iterations for $L = 128$.** Fast enough.

## B.5 Weight Transfer

Standard four-architecture copy. See Section 5.

## B.6 Common Pitfalls

### B.6.1 Forgetting V Masking

Masking K but not V means padding tokens contribute to the state via the 
$KV$ matmul. **Always mask both.**

### B.6.2 Wrong RoPE Extraction

Triple-nested try/except is fragile. Log which signature succeeded.

### B.6.3 The Dead `tau` Parameter

`self.tau` is declared but unused. Remove or implement.

### B.6.4 Cos/Sin Recomputation

Recomputing cos/sin every forward is wasteful. Cache if sequence length 
is fixed.

## B.7 Diagnostics

### Verify Encoder Correctness

```python
# Simple test: L=4, d=2
Q = torch.eye(4)  # one-hot queries
K = torch.eye(4)
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])

# After Cosformer forward
# Check that output is a convex combination of V rows
```

### Verify Causality (Decoder)

```python
x = torch.randn(1, 64, 768, requires_grad=True)
y = model(x)
y[0, 30].sum().backward()
assert x.grad[0, 31:].abs().max() < 1e-6
```

## B.8 Registry

```python
@ATTENTION_REGISTRY.register("cosformer")
class CosformerAttention(BaseAttention): ...
```

## B.9 Injection Factory

Standard pattern. See `05_architecture.md`.

---
# Appendix B. Implementation Details

## B.1 Core Module

```python
class ChelaSlayAttention(BaseAttention):
    def __init__(self, config, rank=64, feature_dim=128, gate_init_bias=5.0, ...):
        super().__init__(config, rank)
        
        self.dim = ...
        self.num_heads = ...
        self.head_dim = ...
        self.feature_dim = feature_dim
        
        # LoRA projections
        self.W_q = LoRALinear(...)
        self.W_k = LoRALinear(...)
        self.W_v = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        # Yat-Kernel RFF
        self.register_buffer("random_proj", 
                             torch.randn(self.head_dim, feature_dim) * 0.02)
        
        # Adaptive gate
        self.gate = nn.Linear(self.dim, self.num_heads)
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.fill_(gate_init_bias)
```

## B.2 Spherical Normalization

```python
def _spherical_normalize(self, x):
    return F.normalize(x, p=2, dim=-1, eps=1e-6)
```

## B.3 Yat-Kernel Feature Map

```python
def _yat_feature_map(self, x):
    if self.random_proj.shape[0] != x.shape[-1]:
        self.random_proj = torch.randn(
            x.shape[-1], self.feature_dim, 
            device=x.device, dtype=x.dtype
        ) * 0.02
    proj = torch.matmul(x, self.random_proj.to(device=x.device, dtype=x.dtype))
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (
        1.0 / (self.feature_dim ** 0.25)
    )
```

## B.4 Causal Linear Attention

```python
# Outer product
outer_prod = k_feat.float().unsqueeze(-1) * v_masked.float().unsqueeze(-2)
# Cumsum
S = torch.cumsum(outer_prod, dim=2)
Z = torch.cumsum(k_feat.float(), dim=2)

num = torch.matmul(q_feat.float().unsqueeze(-2), S).squeeze(-2)
denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True) + 1e-5
linear_out = (num / denom.clamp(min=1e-5)).to(x.dtype)
```

## B.5 Bidirectional Linear Attention

```python
KV = torch.matmul(k_feat.transpose(-2, -1).float(), v_masked.float())
Z = k_feat.float().sum(dim=2, keepdim=True)
num = torch.matmul(q_feat.float(), KV)
denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True) + 1e-5
linear_out = (num / denom.clamp(min=1e-5)).to(x.dtype)
```

## B.6 Gate Fusion

```python
g = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
fused = g * dense_out + (1.0 - g) * linear_out
```

## B.7 Common Pitfalls

### B.7.1 Memory OOM in Causal Mode

The cumsum intermediate is large. Reduce batch size or feature_dim.

### B.7.2 Random Proj Not Moved to Device

Ensure `.to(device=x.device)` when the buffer is on a different device.

### B.7.3 Missing NaN Protection

Always `torch.nan_to_num` after softmax.

### B.7.4 Gate Init Too Small

$b_g < 3$ may allow the linear stream to dominate early, causing instability.

### B.7.5 Wrong Nested Encoder Handling

For RoBERTa, `is_nested_encoder=True` skips `out_proj` because the outer 
`attention.output.dense` handles it.

## B.8 Feature Dim Auto-Adjustment

The `random_proj` buffer is registered with `head_dim` rows. If the 
pretrained attention has a different head_dim than expected, the buffer is 
recreated on the fly. This is safe but inefficient.

---
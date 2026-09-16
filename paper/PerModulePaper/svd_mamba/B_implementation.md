<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The SVDMamba Module Structure

```python
@ATTENTION_REGISTRY.register("svd_mamba")
class SVDMamba(BaseAttention):
    def __init__(self, config, rank=None, d_model=768, d_state=16,
                 d_conv=4, expand=1, global_rank=8, lora_rank=16, ...):
        super().__init__(config, rank)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.inner_dim = expand * d_model
        
        # Attention path
        self.q_proj = SVDLinear(d_model, all_head_size, ...)
        self.k_proj = SVDLinear(d_model, kv_head_size, ...)
        self.v_proj = SVDLinear(d_model, kv_head_size, ...)
        self.attn_out_proj = SVDLinear(all_head_size, d_model, ...)
        
        # SSM path
        self.in_proj = SVDLinear(d_model, 2 * inner_dim, ...)
        self.out_proj = SVDLinear(inner_dim, d_model, ...)
        self.x_proj = SVDLinear(inner_dim, dt_rank + 2 * d_state, ...)
        self.dt_proj = SVDLinear(dt_rank, inner_dim, ...)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=d_conv, groups=inner_dim)
        self.A_log = nn.Parameter(...)
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.dt_bias = nn.Parameter(torch.zeros(inner_dim))
        
        # Global SVD mixing
        self.global_V = nn.Parameter(torch.randn(d_model, global_rank) * 0.02)
        self.global_out = SVDLinear(global_rank, inner_dim, ...)
        
        # Gating (init to silence SSM)
        self.gating = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gating.weight)
        nn.init.constant_(self.gating.bias, -7.0)
```

## B.2 The SVDLinear Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None, lora_rank=16, ...):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def forward(self, x):
        out = ((x @ self.V) * self.S) @ self.U.t() + self.bias
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out * self.output_scale
```

## B.3 The SSM Forward

```python
def ssm_forward(self, x):
    # x: [B, L, inner_dim]
    x_t = x.transpose(1, 2)
    x_pad = F.pad(x_t, (self.d_conv - 1, 0))
    x_conv = self.conv1d(x_pad)[..., :seqlen].transpose(1, 2)
    x_conv = F.silu(x_conv)
    
    x_proj = self.x_proj(x_conv)
    delta = F.softplus(self.dt_proj(x_proj[:, :, :dt_rank]) + self.dt_bias)
    delta = delta.clamp(min=1e-3, max=0.1)
    B_ssm = x_proj[:, :, dt_rank:dt_rank+d_state]
    C_ssm = x_proj[:, :, dt_rank+d_state:]
    
    A = -torch.exp(self.A_log.clamp(min=-20, max=10))
    
    # Chunked scan
    h_running = zeros(B, inner_dim, d_state)
    y_chunks = []
    for c in chunks:
        # ... intra-chunk scan + inter-chunk carry
        y_chunks.append(y_c)
    y = cat(y_chunks, dim=1)
    y = y + self.D * x_conv
    return y * F.silu(z)
```

## B.4 The Gating Module

```python
self.gating = nn.Linear(self.d_model, self.d_model)
nn.init.zeros_(self.gating.weight)
nn.init.constant_(self.gating.bias, -7.0)
```

The gate is per-token, per-channel:
```python
gamma = torch.sigmoid(self.gating(hidden_states))
Y = gamma * ssm_proj + (1.0 - gamma) * attn_out
```

## B.5 Common Pitfalls

### B.5.1 Gate Bias Must Be Negative

If $b_g \geq 0$, the SSM contributes from the start and destabilizes training.

### B.5.2 Conv1d Causal Padding

The convolution must be causal: pad only on the left.

### B.5.3 SSM Scan Clamps

Without clamps on `log_alpha`, `h_chunk`, `decay_inv`, `denom`, the SSM 
produces NaN.

### B.5.4 Nested Encoder Detection

RoBERTa requires `is_nested_encoder = True`.

### B.5.5 SVD Factorization Rank

The rank must satisfy $r < \min(d_\text{in}, d_\text{out})$ for 
compression. If $r = \min$, no compression.

---
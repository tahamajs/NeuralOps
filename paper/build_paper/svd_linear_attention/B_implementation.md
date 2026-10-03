<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The SVDLinear Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_rank=16, ...):
        super().__init__()
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def forward(self, x):
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out
```

## B.2 The Decay Gate

```python
def _init_decay_gate(self):
    with torch.no_grad():
        nn.init.kaiming_uniform_(self.decay_gate.weight, a=math.sqrt(5))
        if self.decay_channel_wise:
            decay_times = torch.exp(torch.linspace(math.log(1.0), math.log(1024.0), self.num_heads))
            decay_times = decay_times.repeat_interleave(self.head_dim)
        else:
            decay_times = torch.exp(torch.linspace(math.log(1.0), math.log(1024.0), self.num_heads))
        bias_target = -torch.log(decay_times)
        self.decay_gate.bias.copy_(bias_target)
```

**Note.** As shown in Appendix A.6, this initialization is **inverted**. 
The intended $\tau_h$ values are $[1, 1024]$, but the actual $\tau_h = 1 + 1/\tau_h^\text{target}$ 
ranges from 1.001 to 2. Heads converge during training.

## B.3 The Chunked Scan

```python
def _forward_decay_gated_linear_kernel(self, q, k, v, hidden_states, mask=None, past_state=None):
    B, H, N, D_k = q.shape
    D_v = v.shape[-1]

    phi_q = F.elu(q.float()) + 1.0
    phi_k = F.elu(k.float()) + 1.0
    v_f = v.float()

    if mask is not None:
        phi_k = phi_k * mask_expanded

    # Decay
    decay_logits = self.decay_gate(hidden_states)
    if self.decay_channel_wise:
        decay_logits = decay_logits.view(B, N, H, D_k).permute(0, 2, 1, 3)
        lambda_decay = compute_exponential_decay(decay_logits).unsqueeze(-1)
    else:
        decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        lambda_decay = compute_exponential_decay(decay_logits)

    # Force padding to have λ=1
    if mask is not None:
        lambda_decay = torch.where(mask_decay > 0, lambda_decay, torch.ones_like(lambda_decay))

    # Chunked scan
    chunk_size = 64
    num_chunks = (N + chunk_size - 1) // chunk_size
    S_chunk = torch.zeros(B, H, D_k, D_v, dtype=torch.float32)
    Z_chunk = torch.zeros(B, H, D_k, 1, dtype=torch.float32)
    outputs = []

    for c in range(num_chunks):
        s, e = c * chunk_size, min(N, (c + 1) * chunk_size)
        qc, kc, vc = phi_q[:, :, s:e], phi_k[:, :, s:e], v_f[:, :, s:e]
        lam_c = lambda_decay[:, :, s:e]
        Lc = e - s

        # Intra-chunk
        log_lam = torch.log(lam_c.clamp(min=1e-6))
        log_gamma = torch.cumsum(log_lam, dim=-1)
        log_M = log_gamma.unsqueeze(-1) - log_gamma.unsqueeze(-2)
        M_intra = torch.exp(log_M.clamp(max=0.0)) * causal_tril
        aff = torch.matmul(qc, kc.transpose(-1, -2)) * M_intra
        num_intra = torch.matmul(aff, vc)
        den_intra = aff.sum(dim=-1, keepdim=True)

        # Inter-chunk
        decay = torch.exp(log_gamma.clamp(max=0.0)).unsqueeze(-1)
        num_inter = torch.matmul(qc, S_chunk) * decay
        den_inter = torch.matmul(qc, Z_chunk) * decay

        out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps)
        outputs.append(out_c)

        # Update state
        total_decay = torch.exp(log_gamma[:, :, -1:].clamp(max=0.0)).unsqueeze(-1)
        w_to_end = torch.exp((log_gamma[:, :, -1:] - log_gamma).clamp(max=0.0)).unsqueeze(-1)
        kc_w = kc * w_to_end
        S_chunk = total_decay * S_chunk + torch.matmul(kc_w.transpose(-1, -2), vc)
        Z_chunk = total_decay * Z_chunk + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2)

    context = torch.cat(outputs, dim=2).to(q.dtype)
    return context, (S_chunk, Z_chunk)
```

## B.4 The `step_recurrent` Method

```python
def step_recurrent(self, x_t, past_state=None):
    if x_t.dim() == 2:
        x_t = x_t.unsqueeze(1)
    
    B = x_t.shape[0]
    q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
    
    if self.num_kv_heads < self.num_heads:
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)
    
    phi_q = F.elu(q.float()) + 1.0
    phi_k = F.elu(k.float()) + 1.0
    
    # Decay
    if self.use_decay_gating:
        decay_logits = self.decay_gate(x_t)
        # ... compute λ_t ...
    else:
        lam_t = torch.ones(B, H, 1, 1)
    
    # Update state
    if past_state is not None:
        S_prev, Z_prev = past_state
    else:
        S_prev = torch.zeros(B, H, D_k, D_v)
        Z_prev = torch.zeros(B, H, D_k, 1)
    
    kt_t = phi_k.transpose(-1, -2)
    kv_t = torch.matmul(kt_t, v.float())
    
    S_t = lam_t * S_prev + kv_t
    Z_t = lam_t * Z_prev + kt_t
    
    num_t = torch.matmul(phi_q, S_t)
    den_t = torch.matmul(phi_t, Z_t) + self.eps
    y_t = num_t / den_t
    
    # Output projection
    context = y_t.transpose(1, 2).contiguous().view(B, 1, self.all_head_size)
    if self.is_nested_encoder:
        out_t = context
    else:
        out_t = self.out_proj(context)
    
    return out_t, (S_t, Z_t)
```

## B.5 Common Pitfalls

### B.5.1 Decay Init Is Inverted

As noted in A.6, the decay bias initialization is inverted. Consider 
fixing to `bias_target = -softplus_inv(log(decay_times))`.

### B.5.2 Padding in Feature Map

φ(x) = ELU(x) + 1 > 0 always, so padding tokens contribute to the state. 
We mask them with `phi_k = phi_k * mask_expanded`.

### B.5.3 Chunk Size

Fixed at 64. Larger chunks give more parallelism but reduce decay effect.

### B.5.4 Nested Encoder Output Projection

For RoBERTa, `is_nested_encoder = True` means the output projection is 
skipped (applied externally).

---
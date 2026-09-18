<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Module Structure

```python
@ATTENTION_REGISTRY.register("oha")
class OptimizedHybridAttention(BaseAttention):
    def __init__(self, config, rank=64, window_size=64,
                 compression_rank=32, use_gate=True, gate_bias=2.0,
                 pretrained_attn=None, ...):
        super().__init__(config, rank)
        self.window_size = window_size
        self.compression_rank = compression_rank
        self.use_gate = use_gate
        
        self.query = LoRALinear(d, H*d_h, lora_rank=16)
        self.key = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.value = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_proj = LoRALinear(H*d_h, d, lora_rank=16)
        
        self.compress_k = nn.Linear(d, compression_rank, bias=False)
        self.compress_v = nn.Linear(d, compression_rank, bias=False)
        
        if use_gate:
            self.gate = nn.Linear(d, H)
            nn.init.constant_(self.gate.bias, gate_bias)
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, N, D = hidden_states.shape
    H, d = self.num_heads, self.head_dim
    
    # Projections
    Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
    K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
    V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
    
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, *position_embeddings)
    
    if self.kv_repeat > 1:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    valid_mask_2d = get_token_valid_mask(attention_mask, B, N, device)
    key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
    
    scale = 1.0 / math.sqrt(d)
    
    # Local branch
    idx = torch.arange(N, device=device)
    if self.is_decoder:
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) >= 0) & \
                    (idx.unsqueeze(1) - idx.unsqueeze(0) <= self.window_size)
    else:
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
    local_mask = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
    
    local_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    local_scores = local_scores.masked_fill(~local_mask, -1e4)
    local_probs = F.softmax(local_scores, dim=-1).to(dtype)
    local_ctx = torch.matmul(local_probs, V)
    
    # Compression
    comp_k_raw = self.compress_k(hidden_states)
    comp_v_raw = self.compress_v(hidden_states)
    mask_inv = (~valid_mask_2d).unsqueeze(-1)
    comp_k_raw = comp_k_raw.masked_fill(mask_inv, -1e4)
    comp_v_raw = comp_v_raw.masked_fill(mask_inv, -1e4)
    
    comp_k_weights = F.softmax(comp_k_raw, dim=1)
    comp_v_weights = F.softmax(comp_v_raw, dim=1)
    
    ck = torch.einsum("bhnd,bnr->bhrd", K, comp_k_weights)
    cv = torch.einsum("bhnd,bnr->bhrd", V, comp_v_weights)
    
    # Global attention
    global_scores = torch.matmul(Q.float(), ck.float().transpose(-2, -1)) * scale
    
    if self.is_decoder:
        causal_global = torch.tril(torch.ones(N, self.compression_rank, device=device, dtype=torch.bool))
        global_scores = global_scores.masked_fill(~causal_global.unsqueeze(0).unsqueeze(0), -1e4)
    
    global_probs = F.softmax(global_scores, dim=-1).to(dtype)
    global_ctx = torch.matmul(global_probs, cv)
    
    # Gating
    if self.use_gate:
        gate = torch.sigmoid(self.gate(hidden_states))
        gate = gate.permute(0, 2, 1).unsqueeze(-1)
        context = gate * local_ctx + (1 - gate) * global_ctx
    else:
        context = 0.5 * local_ctx + 0.5 * global_ctx
    
    context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, None
    return self.out_proj(context), None
```

## B.3 The `raise` on Copy Failure

**Critical.** OHA's `copy_weights` raises `RuntimeError` on missing projections:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

And `inject_oha` also raises:

```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

**Implication.** OHA fails hard on any injection problem. Unlike NHA/EAT, 
there is no silent fallback.

## B.4 The Unused `output_scale`

In `LoRALinear.__init__`:

```python
if not hasattr(self, 'output_scale'):
    self.output_scale = nn.Parameter(torch.ones(1))
```

And in `forward`:

```python
return out * self.output_scale
```

This parameter is initialized to 1.0 and multiplies the output. It is 
unused in practice (no training signal to change it).

## B.5 The `output_scale` in `OptimizedHybridAttention`

Note: `output_scale` is only added in `LoRALinear`, not in the parent 
attention module. So the attention output is `out_proj(context)`, where 
`out_proj` scales by its own `output_scale` (init 1.0).

## B.6 Common Pitfalls

### B.6.1 Forgetting Causal Mask in Global Branch

Without `torch.tril`, the global branch leaks future information.

### B.6.2 Wrong `is_nested_encoder`

RoBERTa requires `True`. Otherwise double projection.

### B.6.3 Missing Compression Mask

Padding tokens must be masked in the compression softmax (`mask_inv`).

### B.6.4 GPT-2 Conv1D Transpose

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

---

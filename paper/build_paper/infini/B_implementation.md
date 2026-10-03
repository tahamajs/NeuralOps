<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Module Structure

```python
@ATTENTION_REGISTRY.register("infini")
class InfiniAttention(BaseAttention):
    def __init__(
        self, config, rank=64, compress_dim=128, local_window=128, pretrained_attn=None, **kwargs
    ):
        super().__init__(config, rank)

        # Dimensions
        self.dim = 768
        self.num_heads = 12
        self.head_dim = 64
        self.all_head_size = 768
        self.compress_dim = 128
        self.local_window = 128

        # Projections with LoRA
        self.q_proj = LoRALinear(768, 768, lora_rank=16)
        self.k_proj = LoRALinear(768, 768, lora_rank=16)
        self.v_proj = LoRALinear(768, 768, lora_rank=16)
        self.out_proj = LoRALinear(768, 768, lora_rank=16)

        # Compressive memory
        self.memory = nn.Parameter(torch.randn(1, 12, 128, 64) * 0.001)
        self.beta = nn.Parameter(torch.tensor([-5.0]))

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, L, D = hidden_states.shape
    H, Dh = self.num_heads, self.head_dim
    
    # 1. Projections
    q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
    
    # 2. RoPE
    if position_embeddings is not None:
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # 3. GQA
    if self.kv_repeat > 1:
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)
    
    # 4. Memory attention
    mem = self.memory.expand(B, -1, -1, -1)  # (B, H, d_c, Dh)
    mem_scores = torch.matmul(q.float(), mem.float().transpose(-2, -1)) / math.sqrt(Dh)
    mem_attn = torch.softmax(mem_scores, dim=-1)
    mem_out = torch.matmul(mem_attn, mem.float()).to(v.dtype)
    
    # 5. Local attention
    scale = 1.0 / math.sqrt(Dh)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    
    # Sliding window mask
    if self.is_decoder or self.is_causal:
        idx = torch.arange(L, device=q.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() < self.local_window
        window_mask = causal & window
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e4)
    
    # Padding mask
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
        elif attention_mask.dim() == 4:
            scores = scores + attention_mask.clamp(min=-1e4)
    
    attn_local = F.softmax(scores, dim=-1).to(v.dtype)
    attn_local = torch.nan_to_num(attn_local, nan=0.0)
    attn_local = self.dropout(attn_local)
    
    local_out = torch.matmul(attn_local, v)
    
    # 6. Gated fusion
    beta = torch.sigmoid(self.beta)
    combined = local_out + beta * mem_out
    combined = combined.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
    
    # 7. Output projection
    if self.is_nested_encoder:
        return combined, None
    out = self.out_proj(combined)
    return out, None
```

## B.3 Memory Initialization

```python
self.memory = nn.Parameter(torch.randn(1, H, d_c, d_h) * 0.001)
```

The scale factor `0.001` ensures the memory's initial output magnitude is 
small, so even with $\sigma(\beta) \approx 0.007$, the contribution is 
negligible.

## B.4 Gate Initialization

```python
self.beta = nn.Parameter(torch.tensor([-5.0]))
```

At $\beta = -5$: $\sigma(-5) \approx 0.0067$, so memory contributes <1% 
initially.

## B.5 Sliding Window Mask

For causal models:

```python
idx = torch.arange(L, device=q.device)
causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # j <= i
window = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() < W
mask = causal & window
```

For bidirectional models, only `window` is used.

## B.6 Common Pitfalls

### B.6.1 Memory Scale Too Large

If `memory` is initialized with `torch.randn(...)` without scaling, the 
initial memory output is large, and even $\sigma(-5) = 0.007$ multiplied by 
it can disrupt the pretrained behavior.

### B.6.2 β Init at 0

At $\beta = 0$, $\sigma(0) = 0.5$. The memory contributes 50% at start, 
disrupting the pretrained local attention.

### B.6.3 Missing Window Mask

Without the window mask, local attention becomes full attention — the 
whole point is lost.

### B.6.4 Nested Encoder Mis-Set

For RoBERTa, `is_nested_encoder = True`. Otherwise double projection.

---
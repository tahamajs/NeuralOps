<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `AdaptiveSpanAttention` Class

```python
class AdaptiveSpanAttention(BaseAttention):
    def __init__(self, config=None, max_span=128, ramp_size=32.0, ...):
        super().__init__(config, rank)
        
        # Resolve dimensions
        ...
        
        # Span logits
        self.span_logits = nn.Parameter(torch.full((H,), 4.0))
        self.max_span = max_span
        self.ramp_size = ramp_size
        
        # Projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)  # or IdentityOutProj for nested
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 Forward Pass (Ramp Application)

```python
# Compute span from logit
span = torch.sigmoid(self.span_logits) * self.max_span  # [H]

# Fast path: skip ramp if unnecessary
if not ((span + self.ramp_size) >= N_k).all():
    pos_q = torch.arange(N_q, device=device, dtype=scores.dtype).unsqueeze(1)
    pos_k = torch.arange(N_k, device=device, dtype=scores.dtype).unsqueeze(0)
    
    if is_decoder:
        dist = (pos_q - pos_k).clamp(min=0.0)
    else:
        dist = (pos_q - pos_k).abs()
    
    span_h = span.view(1, H, 1, 1).to(scores.dtype)
    dist_mat = dist.view(1, 1, N_q, N_k)
    
    ramp = torch.clamp(
        (self.ramp_size + span_h - dist_mat) / self.ramp_size,
        0.0, 1.0
    )
    
    attn = attn * ramp
    attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    attn = attn / attn_sum
    attn = torch.nan_to_num(attn, nan=0.0)
```

## B.3 Span Properties

```python
@property
def current_span(self) -> torch.Tensor:
    return torch.sigmoid(self.span_logits) * self.max_span

def get_span_loss(self) -> torch.Tensor:
    return self.current_span.sum()
```

## B.4 Identity Out Projection for Nested

```python
class IdentityOutProj(nn.Identity):
    lora_A = None
    lora_B = None
    def freeze_base_weights(self):
        pass
```

For nested encoders (RoBERTa), we use this to skip out_proj (the outer 
`attention.output.dense` handles it).

## B.5 Handling Causal Distance

```python
if is_decoder:
    dist = (pos_q - pos_k).clamp(min=0.0)  # Only past keys
else:
    dist = (pos_q - pos_k).abs()  # Both directions
```

## B.6 Common Pitfalls

### B.6.1 Log of Zero

`torch.log(ramp)` returns `-inf` for `ramp = 0`. Add `epsilon = 1e-8`:
```python
attn = attn * ramp  # or: attn = attn + torch.log(ramp + 1e-8)
```

### B.6.2 Wrong Renormalization

After ramp, renormalize:
```python
attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-4)
```

### B.6.3 FP16 Ramp

Compute ramp in FP32, cast back:
```python
ramp = torch.clamp(...).to(scores.dtype)
```

### B.6.4 Initial Logit Too Small

If initialized with `logit = 0` (span ≈ 0.5 * max), accuracy drops from start. 
Use `logit = 4.0`.

### B.6.5 Missing Fast Path

Without fast path, ramp is allocated even for short sequences. Add the 
`if not ((span + ramp) >= N_k).all()` check.

---

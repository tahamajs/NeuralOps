# Appendix B. Implementation Details

## B.1 Module Structure

```python
class FlashAttention3(BaseAttention):
    def __init__(self, config, rank=None, use_fp8=False, pretrained_attn=None, **kwargs):
        super().__init__(config, rank)
        # Resolve dims
        self.hidden_size = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = ...
        self.kv_repeat = ...
        self.use_fp8 = use_fp8
        
        # LoRA projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        # Copy weights
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Dispatch Logic

```python
if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
    try:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        
        if self.use_fp8 and hasattr(torch, "float8_e4m3fn"):
            q_fa = q_fa.to(torch.float8_e4m3fn)
            k_fa = k_fa.to(torch.float8_e4m3fn)
            v_fa = v_fa.to(torch.float8_e4m3fn)
        
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag).to(dtype)
        out = out.contiguous().view(B, L, self.all_head_size)
    except Exception:
        out = None
else:
    out = None

if out is None:
    out = self._sdpa_fallback(q, k, v, attn_mask, causal_flag)
```

## B.3 SDPA Fallback

```python
def _sdpa_fallback(self, q, k, v, attn_mask, causal):
    if causal and attn_mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        if causal:
            causal_m = torch.triu(torch.full((q.shape[-2], k.shape[-2]), float("-inf")), diagonal=1)
            attn_mask = causal_m.unsqueeze(0).unsqueeze(0) if attn_mask is None else attn_mask + causal_m
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

## B.4 Multi-Architecture Weight Extraction

### GPT-2 Conv1D
```python
fused = src.c_attn.weight.T
q_w, k_w, v_w = fused.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### RoBERTa Nested
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
is_nested = True
```

### DistilBERT Flat
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### Qwen GQA
```python
q_w = src.q_proj.weight
k_w = src.k_proj.weight
v_w = src.v_proj.weight
o_w = src.o_proj.weight
num_kv_heads = config.num_key_value_heads
```

## B.5 Common Pitfalls

### B.5.1 Forgetting import fallback
```python
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
```

### B.5.2 Wrong contiguity
FlashAttention requires contiguous tensors in `(B, L, H, d_h)` layout:
```python
q_fa = q.transpose(1, 2).contiguous()
```

### B.5.3 FP8 without Hopper
`torch.float8_e4m3fn` exists on any CUDA, but tensor cores only on Hopper.

### B.5.4 Nested encoder double projection
For RoBERTa: `is_nested_encoder = True` skips `out_proj`.

## B.6 Verification

```python
from framework.attentions.flash_attention3 import FLASH_ATTN_AVAILABLE
print(FLASH_ATTN_AVAILABLE)  # False on RTX 4070

# Exactness test
out_sdpa = F.scaled_dot_product_attention(q, k, v)
out_naive = torch.softmax(q @ k.T / 8.0, -1) @ v
print((out_sdpa - out_naive).abs().max())  # < 1e-5
```

---
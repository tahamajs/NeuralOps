# Appendix B. Implementation Details

## B.1 Module Structure

```python
class FlashAttention3(BaseAttention):
    def __init__(self, config, rank=None, use_fp8=False, ...):
        super().__init__(config, rank)
        
        self.num_heads = ...
        self.head_dim = ...
        self.hidden_size = ...
        self.num_kv_heads = ...
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)
        self.use_fp8 = use_fp8
        
        # LoRA projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        self.rotary_emb = None
        
        if source_attn is not None:
            self.copy_weights(source_attn)
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
```

## B.3 SDPA Fallback

```python
if out is None:
    eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
    attn_mask = None
    if eff_mask is not None:
        if eff_mask.dim() == 2:
            attn_mask = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4
        elif eff_mask.dim() == 3:
            attn_mask = eff_mask.unsqueeze(1).float()
        elif eff_mask.dim() == 4:
            attn_mask = eff_mask.float()
    
    if causal_flag and attn_mask is None:
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, 
            dropout_p=self.dropout.p if self.training else 0.0
        )
    else:
        if causal_flag:
            causal_m = torch.triu(torch.full((L, kv_len), -inf, ...), diagonal=1)
            attn_mask = (attn_mask + causal_m...) if attn_mask is not None else causal_m.unsqueeze(0).unsqueeze(0)
        
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, ...
        )
    
    out = out_sdpa.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
```

## B.4 Nested Encoder Detection

```python
if has_out_lin:
    is_nested = False  # DistilBERT has out_lin → flat
elif (has self and output.dense):
    is_nested = True   # RoBERTa nested
elif cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]:
    is_nested = True
else:
    is_nested = False
```

**Note.** The order matters: `has_out_lin` is checked first.

## B.5 Common Pitfalls

### B.5.1 FA3 Silent Failure

The FA3 dispatch is wrapped in `try/except`; failures fall through to SDPA. 
This is intentional but can silently hide configuration issues.

### B.5.2 FP8 Without Hopper

If `use_fp8=True` but the GPU is not Hopper, the cast may succeed but the 
kernel call fails, falling back to SDPA. The FP8 branch is currently only 
checked by `hasattr(torch, "float8_e4m3fn")`, not by architecture.

### B.5.3 Nested Detection Order

Because `has_out_lin` is checked first, some architectures may be 
misclassified. Specifically, if a nested model has `out_lin` (some 
architectures do), it would be classified as flat.

### B.5.4 Cross-Attention and Causal Flag

`causal_flag = is_decoder and not is_cross_attention and L == kv_len`. This 
is correct: cross-attention is not causal.

## B.6 Fallback Verification

**Figure B.1** compares output from FA3 and SDPA on the same input. Max 
difference is ~1e-3 (FP16 precision).

## B.7 Installation

```bash
pip install flash-attn --no-build-isolation
```

Requires:
- CUDA ≥ 12.0
- PyTorch ≥ 2.2
- Hopper (H100, H200) for FA3

For FA2 (Ampere+):
```bash
pip install flash-attn==2.x
```

## B.8 Recommended Settings

- `use_fp8=False` for safety.
- `lora_rank=16` (standard).
- Test on Hopper before deployment.
```

---
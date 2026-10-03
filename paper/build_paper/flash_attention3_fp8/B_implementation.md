<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 FP8 Import Detection

```python
try:
    from flash_attn_interface import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func

        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
```

Two import paths: `flash_attn_interface` (Hopper-optimized) and `flash_attn` 
(community).

## B.2 FP8 Forward Path

```python
if FLASH_ATTN_AVAILABLE and self.use_fp8 and q.is_cuda and hasattr(torch, "float8_e4m3fn"):
    try:
        q_fa = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        k_fa = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        v_fa = v.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag).to(dtype)
        out = out.contiguous().view(B, L, self.all_head_size)
    except Exception:
        out = None
```

## B.3 SDPA Fallback

```python
if out is None:
    if causal_flag and attn_mask is None:
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, dropout_p=self.dropout.p
        )
    else:
        # Combine causal and padding
        if causal_flag:
            causal_m = torch.triu(torch.full((L, kv_len), float("-inf"), device=device), diagonal=1)
            attn_mask = (
                (attn_mask + causal_m.unsqueeze(0).unsqueeze(0))
                if attn_mask is not None
                else causal_m.unsqueeze(0).unsqueeze(0)
            )
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=self.dropout.p
        )
    out = out_sdpa.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
```

## B.4 Weight Copy (GPT-2)

```python
if hasattr(src, "c_attn"):
    w = src.c_attn.weight.data.clone()
    if src.c_attn.__class__.__name__ == "Conv1D":
        w = w.t()
    q_w, k_w, v_w = w.chunk(3, dim=0)
    extracted["q"] = (q_w, ...)
```

## B.5 Weight Copy (RoBERTa)

```python
elif hasattr(src, "self") and hasattr(src.self, "query"):
    extracted["q"] = (src.self.query.weight.data, ...)
    extracted["k"] = (src.self.key.weight.data, ...)
    extracted["v"] = (src.self.value.weight.data, ...)
    if hasattr(src.output, "dense"):
        extracted["o"] = (src.output.dense.weight.data, ...)
```

## B.6 Weight Copy (DistilBERT)

```python
else:
    extracted["q"] = (src.q_lin.weight.data, ...)
    extracted["k"] = (src.k_lin.weight.data, ...)
    extracted["v"] = (src.v_lin.weight.data, ...)
    extracted["o"] = (src.out_lin.weight.data, ...)
```

## B.7 Nested vs Flat Detection

```python
if hasattr(src, "out_lin"):  # DistilBERT
    is_nested = False
elif hasattr(src, "self") and hasattr(src.self, "query"):  # RoBERTa
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
    is_nested = True
else:
    is_nested = False
```

**Critical:** Check `out_lin` first. DistilBERT's `MultiHeadSelfAttention` 
ends with "SelfAttention" and would be misclassified otherwise.

## B.8 Common Pitfalls

### B.8.1 Missing FP8 Detection

Without detection:

```python
# WRONG — crashes on RTX 4070
out = flash_attn_func(q.to(torch.float8_e4m3fn), ...)

# CORRECT
if FLASH_ATTN_AVAILABLE and hasattr(torch, "float8_e4m3fn"):
    ...
```

### B.8.2 Wrong Conv1D Orientation

```python
# WRONG
w = src.c_attn.weight.data  # (768, 2304) for GPT-2
q_w, k_w, v_w = w.chunk(3, dim=0)  # wrong shapes

# CORRECT
w = src.c_attn.weight.data.t()  # (2304, 768)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### B.8.3 Missing causal_flag

```python
# WRONG — applies causal to cross-attention
causal_flag = self.is_decoder

# CORRECT
causal_flag = self.is_decoder and not is_cross_attention and L == kv_len
```

## B.9 Diagnostics

```python
# Print hardware capabilities
import torch

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Float8 exists:", hasattr(torch, "float8_e4m3fn"))
print("flash_attn available:", FLASH_ATTN_AVAILABLE)

# In forward, add:
logger.info("FP8 path taken: %s", out is not None)
```

## B.10 Registry

```python
@ATTENTION_REGISTRY.register("flash3_fp8")
@ATTENTION_REGISTRY.register("flash_attention3_fp8")
class FlashAttention3FP8(BaseAttention): ...
```

## B.11 Injection Factory

```python
@ATTENTION_REGISTRY.register("flash3_fp8_inject")
def inject_flash3_fp8(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or extract_attention(layer)
    try:
        new_attn = FlashAttention3FP8(config=model_config, old_attn=src_attn)
        replace_attention(layer, new_attn)
        return new_attn
    except Exception as e:
        logger.error(f"Injection failed: {e}")
        return None
```

---
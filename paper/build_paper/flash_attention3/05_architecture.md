# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Cross-attn? |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | No |
| RoBERTa | Nested | Yes | No | No | No |
| GPT-2 | Causal | No | No | No | No |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | No |
| T5 | Enc-Dec | No | No | No | Yes |

## 5.2 Weight Extraction

### DistilBERT
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight  # Only for reference
is_nested_encoder = True
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T  # (2304, 768)
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
```python
q_w = src.q_proj.weight
k_w = src.k_proj.weight
v_w = src.v_proj.weight
o_w = src.o_proj.weight
num_kv_heads = config.num_key_value_heads
```

## 5.3 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True`:

```python
if self.is_nested_encoder:
    final_out = out  # Skip out_proj
else:
    final_out = self.out_proj(out)
```

The outer `BertSelfOutput.dense` applies its projection.

## 5.4 Partial RoPE

For Qwen3.5 with rotary_dim < head_dim:

```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
q = torch.cat([q_rot, q_pass], dim=-1)
```

## 5.5 Causality

```python
causal_flag = bool(self.is_decoder and not is_cross_attention and L == kv_len)
```

- If decoder self-attention with L == kv_len: apply causal.
- Cross-attention: no causal mask.
- Encoder self-attention: no causal mask.

## 5.6 Cross-Attention

For encoder-decoder models:

```python
kv_input = encoder_hidden_states
# No causal masking
```

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("flash3")
@ATTENTION_REGISTRY.register("flash_attention3")
class FlashAttention3(BaseAttention): ...
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| FA3 not installed | ImportError | SDPA fallback |
| FA3 kernel fails | try/except | SDPA fallback |
| FP8 unsupported | hasattr check | FP16 |
| Missing weights | Frobenius check | Warning + skip |
| Shape mismatch | Reshape attempt | Broadcast fallback |

## 5.9 Dispatch Logic

```python
if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
    try:
        out = flash_attn_func(q, k, v, causal=causal_flag)
    except Exception:
        out = None
else:
    out = None

if out is None:
    out = F.scaled_dot_product_attention(q, k, v, ...)
```

## 5.10 Numerical Precision

| Component | Precision |
|---|---|
| Base weights | Model dtype |
| LoRA | Model dtype |
| Attention (FA3) | Model dtype or FP8 |
| Attention (SDPA) | Model dtype |
| Output | Model dtype |

---
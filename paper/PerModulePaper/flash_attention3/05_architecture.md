# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Cross-attn |
|---|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No | No |
| GPT-2 | `h[i].attn` | No | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes | No |
| T5 | `block[i].layer[0]` + `layer[1]` | No | No | No | Yes |

## 5.2 Weight Extraction

Handles four patterns:

### 5.2.1 GPT-2 Conv1D

```python
c_attn_mod.merge()
c_w = c_attn_mod.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
...
o_w = src.output.dense.weight
```

### 5.2.3 Flat Modules

`q_lin`, `q_proj`, `query`, etc.

### 5.2.4 UniversalAdapter Fallback

If any projection is missing.

## 5.3 Nested Encoder Detection

**Note the specific order:**

```python
if has_out_lin:
    is_nested = False
elif (has self and has output.dense):
    is_nested = True
elif cls_name in ["BertSelfAttention", ...]:
    is_nested = True
else:
    is_nested = False
```

The `has_out_lin` check comes first, correctly classifying DistilBERT as 
flat (it has `out_lin`).

## 5.4 GQA Support

Before FA3 call:

```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

FA3 also supports native GQA via different `H` and `H_kv` — this is an 
optional optimization we do not currently use.

## 5.5 RoPE Support

Partial-coordinate slicing for Qwen3.5:

```python
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rot(q_rot) * sin, q_pass], dim=-1)
```

## 5.6 Causality

For decoders:

```python
causal_flag = bool(self.is_decoder and not is_cross_attention and L == kv_len)
```

The flag is passed to `flash_attn_func` (native causal) or to SDPA.

## 5.7 Cross-Attention

For encoder-decoder attention:

```python
is_cross_attention = encoder_hidden_states is not None
kv_input = encoder_hidden_states if is_cross_attention else hidden_states
```

FA3 supports cross-attention; the fallback path applies the encoder mask.

## 5.8 Injection Algorithm

**Algorithm 2:** Inject FlashAttention3
```
Input:  model M, config c
Output: M with attention replaced

1: for each layer:
2:     src ← extract_attention(layer)
3:     try:
4:         new ← FlashAttention3(c, old_attn=src)
5:         replace(layer, new)
6:     except Exception as e:
7:         logger.error(f"Layer failed: {e}")
8:         return None
9: return M
```

## 5.9 Failure Handling

The injection **returns None** on failure (does not raise). This is gentler 
than BAM-UMoE because FlashAttention is exact — failing to inject only 
loses the speedup, not accuracy.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("flash3")
@ATTENTION_REGISTRY.register("flash_attention3")
class FlashAttention3(BaseAttention):
    ...
```

---
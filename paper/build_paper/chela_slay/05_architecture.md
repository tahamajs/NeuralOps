# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

The `copy_weights` procedure handles four patterns:

### 5.2.1 GPT-2 Conv1D

```python
c_w = c_attn.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat Modules

```python
q_w = src.q_lin.weight  # or q_proj or query
```

### 5.2.4 UniversalAdapter

Fallback if none match.

## 5.3 Handling of the RFF Projection

The RFF matrix $\Omega$ has **no pretrained equivalent** — it is initialized 
randomly. This is intentional: random features are theoretically justified 
by Bochner's theorem, and the adaptive gate can down-weight the linear 
stream if $\Omega$ is not useful.

## 5.4 Gate Initialization

The gate is a `Linear(dim, H)` module with:

- Weight zero-initialized.
- Bias = 5.0.

This makes $\sigma(b_g) \approx 0.993$, favoring the dense stream.

## 5.5 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`, so we **do not** apply `out_proj`. 
The outer `attention.output.dense` handles the projection.

## 5.6 GQA Support

For GQA models (Qwen), we expand K, V via `repeat_kv` before both streams.

## 5.7 RoPE Support

Partial-coordinate RoPE:

```python
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rot(q_rot) * sin, q_pass], dim=-1)
```

## 5.8 Causality

For decoders, we apply:

1. **Dense stream:** causal mask on scores.
2. **Linear stream:** cumsum over sequence.

Both are strictly causal.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject CHELA-SLAY into a model
```
Input:  model M, config c
Output: M with attention replaced

1: for each layer with attention:
2:     src ← extract_attention(layer)
3:     if src is None: continue
4:     try:
5:         new ← ChelaSlayAttention(c, pretrained_attn=src)
6:         replace(layer, new)
7:     except Exception:
8:         raise RuntimeError("copy_weights failed")
9: return M
```

## 5.10 Failure Handling

The injection raises `RuntimeError` on any failure. This is strict but 
appropriate because the hybrid attention is complex.

## 5.11 Registry

```python
@ATTENTION_REGISTRY.register("chela_slay")
@ATTENTION_REGISTRY.register("chela_slay_attention")
class ChelaSlayAttention(BaseAttention): ...
```

---
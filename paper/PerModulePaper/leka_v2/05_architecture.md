# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

UniversalAdapter first, then fallbacks:

### 5.2.1 GPT-2 Fused Conv1D

```python
c_w = src.c_attn.weight
if c_w.shape[0] < c_w.shape[1]: c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
k_w = s.key.weight
v_w = s.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat Modules

Standard attributes.

## 5.3 Soft Fallback

After UniversalAdapter fails, we log a warning. But the code still 
raises `RuntimeError` at the end if any required projection is missing — 
this can kill the module.

## 5.4 Landmark Scorer Architecture

LayerNorm over $3F$ features → Linear($3F$, 64) → GELU → Linear(64, 1).

## 5.5 Correction Matrix Shape

$W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$, initialized 
as $I_{d_h}$ per head.

**Note.** For $H = 12, d_h = 64$, this is $12 \times 64 \times 64 = 49,152$ 
parameters per layer.

## 5.6 Local-Global Gate

Per-head scalar $g_h \in (0, 1)$, initialized to $\sigma(-2) \approx 0.12$.

**Note.** The gate is applied **before** the `output_scale` in 
LEKAv1 — here, there is no `output_scale`.

## 5.7 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`. Raw context returned.

## 5.8 Causal Masking

**Global branch.** Masked via landmark positions:

```python
q_pos = torch.arange(N).view(1, 1, N, 1)
k_land_pos = land_idx.unsqueeze(2)
causal_land_mask = (k_land_pos <= q_pos)
attn_weights = attn_weights.masked_fill(~causal_land_mask, -1e4)
```

**Local branch.** Standard causal mask.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject LEKAv2 into a model
```
Input:  model M, config c
Output: M with attention replaced

1: for layer in transformer_layers(M):
2:     src ← extract_attention(layer)
3:     if src is None or is_already_injected: continue
4:     try:
5:         new ← LEKAv2Attention(c, old_attn=src)
6:         replace(layer, new)
7:     except Exception:
8:         log_error; continue
9: return M
```

## 5.10 Double-Injection Guard

```python
if isinstance(src_attn, LEKAv2Attention): return None
if "LEKAv2" in src_attn.__class__.__name__: return None
if "LEKA" in src_attn.__class__.__name__: return None
```

## 5.11 Registry

```python
@ATTENTION_REGISTRY.register("leka_v2")
@ATTENTION_REGISTRY.register("leka_v2_attention")
class LEKAv2Attention(BaseAttention):
    ...
```

## 5.12 Cross-Attention

Not supported.

---
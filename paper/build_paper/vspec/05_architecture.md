# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

UniversalAdapter first, then fallback patterns:

### 5.2.1 GPT-2 Fused Conv1D

```python
c_w = src.c_attn.weight
if c_w.shape[0] < c_w.shape[1]:
    c_w = c_w.t()
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

### 5.2.3 Flat

Standard attributes: `q_lin/q_proj/query`, etc.

## 5.3 Nested Encoder Bypass

For RoBERTa/BERT, we:
1. Set `is_nested_encoder = True`.
2. Skip token routing (preserve full sequence).
3. Use `Po` only as identity initialization.

**Important.** Unlike SOA Fused, we still apply `Po` but it's identity-initialized 
in nested case.

## 5.4 Token Routing Guard

The router sets `s[:, 0] = 1e4` to force `[CLS]` retention. This is 
essential for classification tasks.

## 5.5 RoPE Compatibility

RoPE is applied after projection. Since tokens are routed, we must 
**gather the position embeddings** corresponding to selected indices:

```python
cos_routed = torch.gather(cos_b, 1, idx.unsqueeze(-1).expand(...))
sin_routed = torch.gather(sin_b, 1, idx.unsqueeze(-1).expand(...))
Q, K = apply_rotary_pos_emb(Q, K, cos_routed, sin_routed)
```

## 5.6 Head Gate Application

The head gate multiplies the **attention logits** before softmax:

```python
scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale * hg.float()
```

## 5.7 Scatter Back

For non-routed tokens, we set the attention output to zero. This preserves 
the residual connection's contribution — the non-routed tokens pass 
through unchanged.

## 5.8 Injection Algorithm

**Algorithm 2:** Inject VSpec++ into a model
```
Input:  model M, config c, rank r, lora_rank r_L
Output: M with attention replaced

1: for layer in transformer_layers(M):
2:     src ← extract_attention(layer)
3:     if src is None or is_already_injected: continue
4:     try:
5:         new ← VSpecAttention(c, rank=r, lora_rank=r_L, pretrained_attn=src)
6:         replace(layer, new)
7:     except Exception as e:
8:         log_error; continue
9: return M
```

## 5.9 Double-Injection Guard

```python
if isinstance(src, VSpecAttention):
    return None
if "VSpec" in src.__class__.__name__:
    return None
```

## 5.10 Cross-Attention

Not supported. Raises `NotImplementedError`.

## 5.11 Registry

```python
@ATTENTION_REGISTRY.register("vspec")
@ATTENTION_REGISTRY.register("vspec_attention")
class VSpecAttention(BaseAttention): ...
```

---
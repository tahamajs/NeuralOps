# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

Handles four patterns (UniversalAdapter first, then fallbacks).

### 5.2.1 GPT-2 Conv1D

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

### 5.2.3 Flat Modules

```python
q_w = src.q_lin.weight  # or q_proj or query
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

## 5.3 Soft Fallback

On missing weights, we log a warning and keep Kaiming defaults.

## 5.4 Second Q-K Pair

$W_{Q_2}, W_{K_2}$ are Kaiming-initialized and never copied — the parent 
model has no analogue.

## 5.5 The Bottleneck

$W_\text{down}, W_\text{up}$ are randomly initialized (default PyTorch 
init) and trained. They do not need pretrained weights because they 
represent an **arbitrary** concept space.

**Optional:** Future work could initialize from PCA of activations (see 
Appendix D).

## 5.6 Injection Algorithm

**Algorithm 2:** Inject JambaInfo into a layer
```
Input:  layer, config
Output: layer with attention replaced

1: src ← extract_attention(layer)
2: if src is None: return None
3: try:
4:     new ← JambaInfoBlock(config, old_attn=src)
5:     replace(layer, new)
6:     return new
7: except Exception:
8:     log_error
9:     return None
```

## 5.7 Nested Encoder Handling

For RoBERTa, `is_nested_encoder = True`. The forward pass returns raw 
context, and the outer `output.dense` applies the projection.

## 5.8 Causality

Both attention maps and the SSM are causally masked for decoders.

## 5.9 Cross-Attention

Not supported. JambaInfo is encoder-only or decoder-only.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("jamba_info")
class JambaInfoBlock(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("jamba_info_inject")
def inject_jamba_info(...):
    ...
```

---
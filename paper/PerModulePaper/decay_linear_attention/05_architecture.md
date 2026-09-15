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

Four patterns:

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

Fallback.

## 5.3 GQA Handling

We expand KV heads via `repeat_kv` before the scan:

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.4 RoPE Support

Partial-coordinate slicing for Qwen3.5:

```python
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rot(q_rot) * sin, q_pass], dim=-1)
```

## 5.5 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`. The forward pass returns raw 
context; the outer `attention.output.dense` applies the output projection.

## 5.6 Cross-Attention

For encoder-decoder attention:

```python
is_cross_attention = encoder_hidden_states is not None
kv_input = encoder_hidden_states if is_cross_attention else hidden_states
```

K, V project from `kv_input`; Q from `hidden_states`. The scan operates on 
the cross-attention Q, K, V.

**Important.** In cross-attention, we use **only forward scan** (no 
bidirectional) because the encoder output has already been processed 
bidirectionally.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject DecayLinearAttention
```
Input:  model M, config c
Output: M with attention replaced

1: for each layer with attention:
2:     src ← extract_attention(layer)
3:     if src is None: continue
4:     try:
5:         new ← DecayLinearAttention(c, old_attn=src)
6:         replace(layer, new)
7:     except Exception as e:
8:         logger.error(f"Layer failed: {e}")
9:         return None
10: return M
```

## 5.8 Failure Handling

The injection catches exceptions and **returns None** (skip the layer) 
rather than raising. This is gentler than BAM-UMoE and appropriate because 
linear attention is less sensitive to layer skip.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("decay_linear_attention")
@ATTENTION_REGISTRY.register("decay_linear_attn")
class DecayLinearAttention(BaseAttention):
    ...
```

## 5.10 Compatibility Properties

The module exposes Q/K/V/O under multiple names for framework integration:

```python
@property
def query(self): return self.q_proj
@property
def key(self): return self.k_proj
@property
def value(self): return self.v_proj
@property
def q_lin(self): return self.q_proj
...
```

---

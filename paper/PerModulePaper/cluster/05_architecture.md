# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | ✅ |
| RoBERTa | Nested | Yes | No | No | ✅ |
| GPT-2 | Causal | No | No | No | ✅ |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | partial |

## 5.2 Weight Extraction

### DistilBERT (Flat)
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
is_nested_encoder = False
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
# output.dense applied by outer module
is_nested_encoder = True
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
Direct copy with GQA-aware dims.

## 5.3 Injection

```python
def inject_cluster(layer, cfg, comp_config, old_attn):
    new = ClusterAttention(
        config=cfg,
        comp_config=comp_config,
        pretrained_attn=old_attn,
        num_clusters=16,
        n_iters=3,
        window_size=32,
    )
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new
        else:
            layer.attention = new
    return new
```

## 5.4 Causal Masking

For decoders, ClusterFormer uses **exact causal attention**, not clustering:

```python
if is_decoder:
    scores = Q @ K.T / sqrt(d_h)
    scores = scores.masked_fill(~causal_tril, -10000)
    attn = softmax(scores)
    out = attn @ V
```

**Rationale.** Clustering introduces a global mixing effect that can leak 
future information even with causal masking on keys. For stability we 
use standard exact attention for decoder paths.

## 5.5 RoPE

Partial RoPE (Qwen3.5 with rotary_dim < head_dim) is supported via slicing 
the first `rotary_dim` coordinates.

## 5.6 GQA

```python
if self.kv_repeat > 1:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("cluster")
@ATTENTION_REGISTRY.register("cluster_attention")
class ClusterAttention(BaseAttention):
    ...
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Empty clusters | `counts == 0` | Mask logit to -inf |
| NaN in cdist | `torch.nan_to_num` | Replace with 0 |
| Zero Frobenius | Check < 1e-4 | Raise error |

## 5.9 Numerical Stability

| Component | Precision |
|---|---|
| cdist (K-means) | FP32 |
| Centroid scores | FP32 then cast |
| Softmax | FP32 |
| Attention output | Model dtype |

---
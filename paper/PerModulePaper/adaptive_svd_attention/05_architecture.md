# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? | Relative? | Cross? |
|---|---|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No | No | No |
| GPT-2 | `h[i].attn` | No | No | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes | No | No |
| T5 | `block[i].layer[0].SelfAttention` | No | No | No | Yes | Yes (layer[1]) |

## 5.2 Weight Extraction

Our `copy_weights` handles five patterns:

### 5.2.1 UniversalAdapter First

```python
from framework.core.adapters import UniversalAdapter
w, b = UniversalAdapter.get_weight_bias(src, proj)  # proj ∈ {"q", "k", "v", "o"}
```

Handles most standard architectures.

### 5.2.2 GPT-2 Conv1D

```python
fused_w = src.c_attn.weight.T
fused_b = src.c_attn.bias
q_w, k_w, v_w = fused_w.chunk(3, dim=0)
```

### 5.2.3 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
k_w = s.key.weight
v_w = s.value.weight
o_w = src.output.dense.weight
```

### 5.2.4 DistilBERT Flat

```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### 5.2.5 Fallback `<proj>_proj`

```python
mod = getattr(src, f"{proj}_proj")
w = mod.weight
```

## 5.3 Handling `W_O` Factor

Unlike SOA Fused Attention, we **do not skip $W_O$**. We apply the same 
factorization procedure, with the residual adapter absorbing approximation 
error.

For nested encoders, the outer `attention.output.dense` is **replaced** with 
our factorized `out_proj`. The `is_nested_encoder` flag remains for the 
forward pass, but the output projection is applied inside our module.

## 5.4 T5 Relative Bias Transfer

```python
if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

In `compute_bias`, we generate the position bias for any (query, key) pair.

## 5.5 Cross-Attention

The forward pass handles cross-attention by inspecting `key_value_states`:

```python
key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
is_cross_attention = key_value_states is not None
kv_input = key_value_states if is_cross_attention else hidden_states
```

K, V are computed from `kv_input`; Q from `hidden_states`. The factorized 
projections apply identically.

## 5.6 Causal Detection

Robust prefix matching:

```python
_decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                     "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
is_dec = any(model_type.startswith(p) for p in _decoder_prefixes)
```

Also checks `config.is_decoder`, `config.is_causal`, `config.causal`, and 
the source attention's attributes.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject Adaptive SVD Attention into a model
```
Input:  model M, config c, svd_rank, residual_rank, energy_threshold
Output: M with attention replaced

1: for each layer with attention:
2:     src ← extract_attention(layer)
3:     if src is None or is_already_injected: continue
4:     try:
5:         new ← AdaptiveSVDAttention(c, ..., pretrained_attn=src)
6:         replace(layer, new)
7:     except Exception as e:
8:         log_error(f"Layer failed: {e}")
9:         continue
10: return M
```

## 5.8 Failure Handling

Two-stage error handling:

**Stage 1 — Pre-flight test:** Construct the module inside a try/except. If 
construction fails (weight extraction issue), return `None` and skip.

**Stage 2 — Frobenius validation:** If construction succeeds but a 
projection has near-zero norm, raise `RuntimeError`.

```python
try:
    test_module = AdaptiveSVDAttention(..., pretrained_attn=old_attn)
except Exception:
    return None
```

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_svd_attn")
@ATTENTION_REGISTRY.register("adaptive_svd")
@ATTENTION_REGISTRY.register("adaptive_svd_attention")
class AdaptiveSVDAttention(BaseAttention):
    ...
```

## 5.10 Double-Injection Guard

```python
if isinstance(src_attn, AdaptiveSVDAttention): return None
if "AdaptiveSVD" in src_attn.__class__.__name__: return None
```

---
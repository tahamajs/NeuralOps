# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Transfer

The `copy_weights` procedure handles four patterns, delegated to the two 
branch modules:

### 5.2.1 DistilBERT

```python
q_w = src.q_lin.weight  # (768, 768)
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.t()  # transpose
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.4 Qwen3.5 GQA

```python
q_w = src.q_proj.weight  # (H·d_h, d)
k_w = src.k_proj.weight  # (H_kv·d_h, d)
v_w = src.v_proj.weight
o_w = src.o_proj.weight
```

## 5.3 The `out_proj` Search

The output projection is extracted using a search over candidate attributes:

```python
for cand in ("out_proj", "out_lin", "o_proj", "output", "proj", "c_proj", "dense"):
    for target in (old_attn, old_attn.self, old_attn.attention, old_attn.output):
        if target and hasattr(target, cand):
            ...
```

If no output projection is found, we **raise RuntimeError** to prevent 
injecting a random-init projection.

## 5.4 Injection Algorithm

**Algorithm 2:** Inject HGRN2-GatedDeltaNet into a full model
```
Input:  model M, config c, lora_rank, lora_alpha
Output: M with attention replaced

1: layer_list ← find_transformer_layers(M)
2: for i, layer in enumerate(layer_list) do
3:     src ← extract_attention(layer)
4:     if src is None or is_already_injected(src): continue
5:     try:
6:         new ← HGRN2GatedDeltaNet(c, old_attn=src, ...)
7:         replace(layer, new)
8:     except Exception as e:
9:         log_warning(f"Layer {i} failed: {e}")
10:        continue
11: end for
12: return M
```

## 5.5 The `sub_kwargs` Pattern

To avoid the "multiple values for keyword argument" error, we filter 
`kwargs` before passing to sub-branches:

```python
sub_kwargs = {
    k: v
    for k, v in kwargs.items()
    if k not in ("is_decoder", "is_causal", "lora_rank", "lora_alpha")
}
```

This is essential when the same `kwargs` dict is passed through multiple 
levels of the call hierarchy.

## 5.6 Causal Detection

```python
_decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon")
if any(model_type.startswith(p) for p in _decoder_prefixes):
    is_dec = True
```

The flag is passed to both branches, which use it to switch between 
causal and bidirectional recurrences.

## 5.7 Double-Injection Guard

The injection function checks whether the target already contains a 
`HGRN2GatedDeltaNet`:

```python
if isinstance(curr_attn, HGRN2GatedDeltaNet):
    return None
if "HGRN2GatedDeltaNet" in curr_attn.__class__.__name__:
    return None
if hasattr(curr_attn, "inner") and isinstance(curr_attn.inner, HGRN2GatedDeltaNet):
    return None
if hasattr(curr_attn, "_wrapped_attention") and isinstance(
    curr_attn._wrapped_attention, HGRN2GatedDeltaNet
):
    return None
```

This prevents accidentally wrapping the module twice.

## 5.8 Skipping Hybrid Layers (Qwen3.5)

Qwen3.5 has some layers using `DeltaNet` or `LinearAttention`. We skip them:

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.9 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No Q/K/V weights | `_unpack_weight_bias` returns None | Raise |
| No O weights | Search loop fails | Raise |
| Both branches fail | Both `copy_weights` raise | Raise |
| One branch fails | One `copy_weights` raises | Bias gate |
| Cross-attention | `encoder_hidden_states is not None` | Log warning, return identity |

---
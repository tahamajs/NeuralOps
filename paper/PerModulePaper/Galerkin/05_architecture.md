# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Path |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | **Galerkin** |
| RoBERTa | Nested | Yes | No | No | **Galerkin** |
| GPT-2 | Causal | No | No | No | Fallback |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | Fallback |

## 5.2 Weight Extraction

### DistilBERT
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
is_nested = False
```

### RoBERTa
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
is_nested = True
src = src.self  # Nested: work with inner attention
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen
Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 5.3 Nested Encoder Detection

```python
if (
    hasattr(src, "self")
    and hasattr(getattr(src, "self"), "query")
    and hasattr(src, "output")
    and hasattr(getattr(src, "output"), "dense")
):
    is_nested = True
    src = src.self
```

For nested encoders, the outer `output.dense` applies the projection. We 
return raw `context` and bypass `out_proj`.

## 5.4 GQA Support

```python
if self.num_kv_heads != self.num_heads:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 5.5 RoPE

Applied before the Galerkin projection step. Partial RoPE supported via 
coordinate slicing.

## 5.6 Decoder Detection

```python
_decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                     "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
is_dec = (
    getattr(config, "is_decoder", False)
    or getattr(config, "is_causal", False)
    or any(model_type.startswith(p) for p in _decoder_prefixes)
)
```

## 5.7 Injection

```python
def inject_galerkin(layer, cfg, comp_config, old_attn):
    try:
        new = GalerkinAttention(
            config=cfg,
            comp_config=comp_config,
            old_attn=old_attn,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new
            else:
                layer.attention = new
        return new
    except Exception:
        return None  # Safe fallback
```

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention")
class GalerkinAttention(BaseAttention):
    ...
```

## 5.9 Failure Modes

| Failure | Response |
|---|---|
| Missing weights | logger.warning, continue |
| Zero Frobenius | logger.warning, continue |
| DCT construction error | try/except, fallback to identity basis |
| Injection error | return None |

## 5.10 Numerical Precision

| Component | Precision |
|---|---|
| Base weights | Model dtype |
| LoRA | Model dtype |
| DCT basis | FP32 |
| Basis attention | FP32 for scores |
| Output | Model dtype |

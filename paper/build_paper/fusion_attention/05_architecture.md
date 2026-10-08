# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Cross-attn? |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | No |
| RoBERTa | Nested | Yes | No | No | No |
| GPT-2 | Causal | No | No | No | No |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | No |
| T5 | Enc-Dec | No | No | No | Yes |

## 5.2 Weight Extraction

### DistilBERT (Flat)
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
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
def inject_fusion_attn(layer, cfg, comp_config, old_attn):
    new = FUSIONATTN(
        config=cfg,
        comp_config=comp_config,
        old_attn=old_attn,
        block_size=64,
        top_k_blocks=4,
    )
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new
        else:
            layer.attention = new
    return new
```

## 5.4 Nested Encoder Bypass

For RoBERTa, `out_proj` is not applied; outer `BertSelfOutput.dense` does it.

## 5.5 GQA

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.6 RoPE

Supports partial RoPE (Qwen3.5 with rotary_dim < head_dim).

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("fusion_attn")
class FUSIONATTN(BaseAttention): ...
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Frobenius < 1e-4 | **raise RuntimeError** |
| Zero Frobenius | Same | **raise RuntimeError** |
| Empty blocks | Padding mask | -inf |
| NaN in scores | nan_to_num | Replace 0 |

**⚠️ Critical:** `inject_fusion_attn` has `raise RuntimeError("copy_weights failed")` 
which kills the injection on any failure. Recommend changing to `return None`.

## 5.9 Numerical Stability

| Component | Precision |
|---|---|
| Base weights | Model dtype |
| LoRA | Model dtype |
| Gate MLP | Model dtype |
| Attention scores | FP32 |
| Softmax | FP32 |
| Output | Model dtype |

---
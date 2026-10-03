# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer

AEGIS-V5 copies base weights via `copy_weights`, handling four patterns:

### 5.2.1 GPT-2 Conv1D

```python
weight = old_attn.c_attn.weight.data
if weight.shape[1] == 3 * self.hidden_size:
    weight = weight.t()
q_w, k_w, v_w = weight.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    q_w = old_attn.self.query.weight
    k_w = old_attn.self.key.weight
    v_w = old_attn.self.value.weight
    o_w = old_attn.output.dense.weight
```

### 5.2.3 Qwen GQA

```python
q_w = old_attn.q_proj.weight
k_w = old_attn.k_proj.weight  # (H_kv · d_h, d)
v_w = old_attn.v_proj.weight
o_w = old_attn.o_proj.weight
```

### 5.2.4 DistilBERT

```python
q_w = old_attn.q_lin.weight
k_w = old_attn.k_lin.weight
v_w = old_attn.v_lin.weight
o_w = old_attn.out_lin.weight
```

## 5.3 Robustness Improvements

Compared to AEGIS V4, V5 makes three robustness changes:

### 5.3.1 copy_weights No Longer Raises

In V4, exceptions in `copy_weights` aborted the entire attention replacement. 
V5 logs a warning and continues to fallback paths:

```python
try:
    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
except Exception as e:
    logger.debug(f"UniversalAdapter failed for {proj}: {e}")
    continue
```

### 5.3.2 Frobenius Check No Longer Raises

Near-zero norms trigger a warning, not an abort:

```python
if norm < 1e-4:
    logger.warning(f"Weight for {proj_key} has near-zero norm; continuing")
```

### 5.3.3 inject_aegis_v5 Returns None on Failure

```python
try:
    new_attn.copy_weights(src_attn)
except Exception as e:
    logger.error(f"inject_aegis_v5: copy_weights failed: {e}")
    return None
```

## 5.4 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`:

```python
if self.is_nested_encoder:
    return out, None
return self.out_proj(out), None
```

The outer `BertSelfOutput.dense` applies the output projection.

## 5.5 Decoder Fallback

For decoder models, we use standard causal attention (no split-head):

```python
if N < self.cfg.local_window or self.is_decoder:
    # Standard causal softmax
    scores = Q @ K^T / √d_h
    scores = mask_causal(scores)
    attn = softpick(scores)
    out = attn @ V
    return out, None
```

This preserves causality at the cost of quadratic complexity.

## 5.6 Softpick Fix

The original AEGIS used a ReLU-based softpick that returned all zeros when 
$\max(x) \leq 0$ (e.g., on fully-masked rows). This caused silent attention 
collapse. V5 replaces it with a stable softmax:

```python
def softpick(x, dim=-1, temperature=1.0):
    x_f = x.float()
    if temperature <= 0 or temperature == 1.0:
        return F.softmax(x_f, dim=dim).to(x.dtype)
    return F.softmax(x_f / temperature, dim=dim).to(x.dtype)
```

This fix alone likely prevented several collapse cases.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject AEGIS-V5 into a model
```
Input:  model M, config
Output: M with attention replaced or unchanged on failure

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     cfg ← AegisConfigV5(...)
4:     new ← AegisAttention(model_config, aegis_cfg=cfg)
5:     try:
6:         new.copy_weights(src)
7:     except:
8:         logger.error(...)
9:         continue  # Skip this layer, do not crash
10:    replace(ℓ, new)
11: end for
12: return M
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| UniversalAdapter throws | try/except | Continue to next pattern |
| Missing Q/K/V | `missing` non-empty | Log warning, random init |
| Near-zero norm | Frobenius check | Log warning, continue |
| copy_weights fails | try/except | Skip layer, return None |
| NaN in scores | `nan_to_num` | Set to 0 |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("aegis_v5_module")
@ATTENTION_REGISTRY.register("AegisAttention")
@ATTENTION_REGISTRY.register("aegis_v5")
class AegisAttention(BaseAttention): ...
```

## 5.10 Interaction with LoRA Recovery

Base weights are frozen; LoRA adapters train. Additionally trainable:
- `rff_proj`: the RFF projection matrix
- `blend_logits`: per-head gate logits

For $d = 768$, $H = 12$, $F = 128$:

| Trainable | Params |
|---|---|
| LoRA on 4 projections | ~98K |
| RFF proj | $4 \cdot 64 \cdot 128 = 32,768$ |
| blend_logits | 12 |
| **Total** | **~131K per layer** |

---
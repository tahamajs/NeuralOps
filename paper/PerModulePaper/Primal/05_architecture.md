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

| Architecture | Attention class | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | Yes | Yes | not tested |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

```python
q_lin: Linear(768, 768)
k_lin: Linear(768, 768)
v_lin: Linear(768, 768)
out_lin: Linear(768, 768)
```

**Copy:** Direct. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

```python
attention.self.query: Linear(768, 768)
attention.self.key: Linear(768, 768)
attention.self.value: Linear(768, 768)
attention.output.dense: Linear(768, 768)
```

**Copy:** Extract from inner `self`. `is_nested_encoder = True`, skip own 
`out_proj`.

### 5.2.3 GPT-2 (Causal)

```python
attn.c_attn: Conv1D(768, 2304)   # fused Q, K, V
attn.c_proj: Conv1D(768, 768)
```

**Copy:** Transpose and split. Direct for `c_proj`.

### 5.2.4 Qwen3.5 (GQA + RoPE)

```python
q_proj: Linear(1024, 2048)   # 8 heads × 256
k_proj: Linear(1024, 512)    # 2 heads × 256
v_proj: Linear(1024, 512)
o_proj: Linear(2048, 1024)
```

**Copy:** Direct, then `repeat_kv` at runtime. RoPE applied to landmarks 
implicitly via the pooled $Q, K$ (see §5.5).

## 5.3 Injection Algorithm

**Algorithm 2:** Inject Primal Attention into a full model
```
Input:  model M, config c, m (landmark count)
Output: M with Primal attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← PrimalAttention(config=c, pretrained_attn=src)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.4 Handling RoPE

RoPE is applied to $Q, K$ **before** landmark pooling. This preserves the 
relative positions of the landmark summaries.

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
Q_land, K_land = pool_to_landmarks(Q, K, m)
```

## 5.5 Handling GQA

For grouped-query attention, we repeat KV heads **before** pooling:

```python
if num_kv_heads < num_heads:
    K = repeat_kv(K, num_heads // num_kv_heads)
    V = repeat_kv(V, num_heads // num_kv_heads)
```

Then landmarks are computed on the repeated KV.

## 5.6 Causality

For decoder models, causal masking is applied to the **approximate** attention 
matrix:

```python
causal_mask = torch.triu(torch.full((L, L), -1e4), diagonal=1)
attn = attn + causal_mask
```

**Note.** This is not a true causal approximation (landmarks see all tokens), 
but empirically works for generation. See §8.5 for analysis.

## 5.7 Failure Handling

`inject_primal` returns `None` on failure and logs the error:

```python
try:
    new_attn = PrimalAttention(config=config, pretrained_attn=target, **kwargs)
except Exception:
    raise RuntimeError("copy_weights failed")
```

The outer loop can catch this and continue with the next layer.

## 5.8 Parameter Count

Primal Attention has the **same parameter count** as MHA:

| Projection | Params |
|---|---|
| $W_Q$ | $d \cdot H d_h$ |
| $W_K$ | $d \cdot H_{kv} d_h$ |
| $W_V$ | $d \cdot H_{kv} d_h$ |
| $W_O$ | $H d_h \cdot d$ |

With LoRA recovery at rank $r_L$, add $2 r_L (d + d_h H)$ per projection.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("primal")
@ATTENTION_REGISTRY.register("primal_attention")
class PrimalAttention(BaseAttention): ...
```

## 5.10 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| $F_2$ singular | `linalg.pinv` error | Skip layer |
| NaN in Schulz | `torch.isnan` | Reset to SVD pinv |
| CUDA OOM | RuntimeError | Reduce $m$ |

---

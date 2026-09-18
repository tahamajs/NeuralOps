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

| Architecture | Attention | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | Flat | No | No | ✅ |
| RoBERTa | Nested | No | No | ✅ |
| GPT-2 | Causal | No | No | ✅ |
| Qwen3.5 | Causal + RoPE | Yes | Yes | not tested |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_proj`.

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct copy. `repeat_kv` for GQA. RoPE applied to Q, K.

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, HALO-ReGLA is injected into `attention.self`. The 
`is_nested_encoder = True` flag ensures:
- No `out_proj` is applied inside HALO-ReGLA.
- The outer `attention.output.dense` applies the output projection.

## 5.4 Causality

For decoders, causal masking is applied in the **dense stream** via:

```python
causal_mask = torch.triu(torch.full((L, L), -1e4), diagonal=1)
dense_scores = dense_scores + causal_mask
```

For the **linear stream**, causality is enforced by the prefix sum.

## 5.5 Padding Masking

Both streams handle padding:
- Dense: `masked_fill` with `-inf`.
- Linear: zero out invalid K, V before cumsum.

## 5.6 RoPE Integration

RoPE is applied to Q, K **before** both streams:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
```

Then dense and linear streams process the rotated Q, K.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject HALO-ReGLA into a full model
```
Input:  model M, config c
Output: M with HALO-ReGLA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← HALOReGLAAttention(config=c, pretrained_attn=src)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_halo_regla` returns `None` on failure:

```python
try:
    new_layer = HALOReGLAAttention(...)
except Exception:
    raise RuntimeError("copy_weights failed")
```

## 5.9 Parameter Count

| Projection | Params |
|---|---|
| $W_Q$ | $d H d_h$ |
| $W_K$ | $d H_{kv} d_h$ |
| $W_V$ | $d H_{kv} d_h$ |
| $W_O$ | $H d_h d$ |
| Gate | $d H + H$ |
| LoRA | $2 r (\cdots)$ |

Total: same as MHA + gate (negligible).

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("halo_regla")
@ATTENTION_REGISTRY.register("halo_regla_attention")
class HALOReGLAAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` logs debug | Uses partial init |
| CUDA OOM in cumsum | RuntimeError | Reduce batch |
| NaN in cumsum | `torch.isnan` | Cast FP32 |
| RoPE shape mismatch | Exception | Skip RoPE |

---

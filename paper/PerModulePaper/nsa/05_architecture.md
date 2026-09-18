<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real architectures
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | Flat | No | No | ✅ |
| RoBERTa | Nested | No | No | 0.85 |
| GPT-2 | Causal | No | No | 0.85 |
| Qwen3.5 | Causal + RoPE | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_proj`.

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct copy with `repeat_kv`.

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, `is_nested_encoder = True`:
- No `out_proj` applied inside NSA.
- Outer `attention.output.dense` applies output.

## 5.4 Causality

For decoders:
1. Local mask: $(i - j \leq w) \land (j \leq i)$.
2. Top-$k$ on causal-masked scores.
3. Final AND with causal.

## 5.5 Padding Handling

Padding keys are excluded from both local window and top-$k$:

```python
key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
non_local_scores = non_local_scores.masked_fill(~key_valid_4d, -1e4)
```

## 5.6 RoPE Integration

RoPE applied to Q, K **before** score computation and top-$k$ selection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject NSA into a full model
```
Input:  model M, config c, w, k
Output: M with NSA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← NativeSparseAttention(config=c, pretrained_attn=src, w, k)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_nsa` returns `None` on failure:

```python
try:
    new_attn = NativeSparseAttention(...)
except Exception as e:
    logger.error(f"NSA injection failed: {e}")
    return None
```

## 5.9 Parameter Count

| Component | Baseline | NSA |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | Standard | Same |
| Extra | 0 | **0** |
| LoRA | 0 | $2 r (\cdots)$ |

**NSA adds zero parameters.**

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("nsa")
@ATTENTION_REGISTRY.register("nsa_attention")
class NativeSparseAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| Wrong `is_nested_encoder` | Top-k shape mismatch | Fallback |
| OOM in topk | RuntimeError | Reduce k |
| NaN in scores | `torch.isnan` | Cast FP32 |

---
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
| RoBERTa | Nested | No | No | ✅ |
| GPT-2 | Causal | No | No | ✅ |
| Qwen3.5 | Causal + RoPE | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_lin`.

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct copy with `repeat_kv` at runtime.

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, `is_nested_encoder = True`:
- No `out_lin` applied inside PTP.
- Outer `attention.output.dense` applies output projection.

## 5.4 Causality

For decoders:
1. Score network and top-$k$ ignore causality (they see the full sequence).
2. Attention mask enforces causality by comparing original positions.

**Note.** The score network sees **future** tokens when scoring, which is 
acceptable in non-causal inference (e.g., encoder pretraining) but may 
need adjustment for autoregressive generation.

## 5.5 Padding Handling

Padding tokens are excluded in three places:
1. Score network: $s_i \leftarrow -10^4$.
2. K, V gather: padding never selected.
3. Attention mask: gather valid keys for masking.

## 5.6 GQA Handling

For Qwen/LLaMA with $H_{kv} < H$, we repeat KV heads before pruning:

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    k_all = repeat_kv(k_all, repeats)
    v_all = repeat_kv(v_all, repeats)
```

## 5.7 Injection Algorithm

**Algorithm 2:** Inject PTP into a full model
```
Input:  model M, config c, ρ
Output: M with PTP attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← ProgressiveTokenPruning(config=c, pretrained_attn=src, keep_ratio=ρ)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_progressive_token_pruning` returns `None` on failure:

```python
try:
    new_attn = ProgressiveTokenPruning(...)
except Exception:
    raise RuntimeError("copy_weights failed")
```

## 5.9 Parameter Count

| Component | Baseline | PTP |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | $4 d^2$ | Same |
| Score network | 0 | $64 d$ |
| LoRA | 0 | $2 r_L \cdot 4 d$ |
| **Trainable** | $4 d^2$ | **$64 d + 8 r_L d$** |

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("ptp")
@ATTENTION_REGISTRY.register("progressive_token_pruning")
class ProgressiveTokenPruning(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| `k > L` | Clamped to L | Skip pruning |
| Empty mask | Empty valid_mask | Use all tokens |
| NaN scores | `torch.isnan` | Fallback to uniform |
| Wrong nested flag | Shape mismatch | Fallback |

---
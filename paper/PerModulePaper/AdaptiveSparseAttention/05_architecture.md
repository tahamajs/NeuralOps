<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real architectures
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | ✅ |
| RoBERTa | Nested | No | No | No | ✅ |
| GPT-2 | Causal | No | No | No | ✅ |
| T5 | Causal + bias | No | No | Yes | not tested |
| Qwen3.5 | Causal + RoPE | Yes | Yes | No | not tested |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. The outer `attention.output.dense` is wrapped 
with LoRALinear to allow training:

```python
if hasattr(target.attention, "output"):
    outer_dense = target.attention.output.dense
    if not isinstance(outer_dense, LoRALinear):
        wrapped = LoRALinear(...)
        wrapped.weight.data.copy_(outer_dense.weight.data)
        target.attention.output.dense = wrapped
```

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D → (out, in)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 T5 (Relative Position Bias)

T5 uses relative position bias embeddings. We preserve them:

```python
if hasattr(src, "relative_attention_bias"):
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

The bias is added to scores before the top-$k$ threshold.

### 5.2.5 Qwen3.5 (GQA + RoPE)

Direct copy with `repeat_kv` at runtime. RoPE applied before chunking.

## 5.3 Injection Algorithm

**Algorithm 2:** Inject ASA into a full model
```
Input:  model M, config c
        ρ_enc, ρ_dec (default sparsity)
Output: M with ASA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      ρ_0 ← ρ_dec if is_decoder(src) else ρ_enc
5:      new ← AdaptiveSparseAttention(config=c, pretrained_attn=src, sparsity_ratio=ρ_0)
6:      replace_attention(ℓ, new)
7: end for
```

## 5.4 Handling Nested Encoder

For RoBERTa/BERT, ASA is injected into `attention.self`. The outer 
`attention.output.dense` is wrapped separately with LoRA:

- **Inner ASA** processes Q, K, V and returns raw context (no out_proj).
- **Outer LoRALinear** applies output projection with trainable LoRA.

## 5.5 RoPE Handling

RoPE is applied to Q, K **before** chunking. This preserves relative 
position information in the sparse scores.

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
for q_start in range(0, L, C):
    # ... chunk processing
```

## 5.6 Relative Position Bias (T5)

T5 uses a learned bias added to scores:

$$
s_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_h}} + b_{i-j}
\tag{5.1}
$$

where $b$ is a learned embedding of bucketed relative positions. ASA preserves 
this by computing the bias before the top-$k$ threshold.

## 5.7 Safe Injection

`inject_adaptive_sparse` returns `None` on failure:

```python
try:
    new_block = AdaptiveSparseAttention(...)
except Exception as e:
    logger.error(f"inject_adaptive_sparse failed: {e}")
    return None
```

The caller can then skip the layer.

## 5.8 Handling Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| RoPE embedding missing | Exception | Fallback to no RoPE |
| LoRA rank mismatch | Shape check | Zero-init LoRA |
| CUDA OOM in chunk | RuntimeError | Reduce chunk size |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_sparse")
@ATTENTION_REGISTRY.register("adaptive_sparse_attention")
class AdaptiveSparseAttention(BaseAttention): ...
```

## 5.10 LoRA Integration

LoRA is used for recovery:

| Projection | LoRA rank | Notes |
|---|---|---|
| $W_Q$ | $r_L$ | Full LoRA |
| $W_K$ | $r_L$ | Full LoRA |
| $W_V$ | $r_L$ | Full LoRA |
| $W_O$ (nested) | $r_L$ | Wrapped outer dense |
| Sparsity logits | N/A | Directly trainable |

---
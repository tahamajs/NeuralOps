<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
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

```python
q_lin: Linear(768, 768)
k_lin: Linear(768, 768)
v_lin: Linear(768, 768)
out_lin: Linear(768, 768)
```

Direct copy. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

```python
attention.self.query: Linear(768, 768)
attention.self.key: Linear(768, 768)
attention.self.value: Linear(768, 768)
attention.output.dense: Linear(768, 768)
```

Extract from inner `self`. The outer `attention.output.dense` is preserved 
via `IdentityOutProj` (identity, no operation).

### 5.2.3 GPT-2 (Causal)

```python
attn.c_attn: Conv1D(768, 2304)   # fused Q, K, V
attn.c_proj: Conv1D(768, 768)
```

Transpose `c_attn` weight, split into Q/K/V.

### 5.2.4 T5 (Relative Position Bias)

T5 has `relative_attention_bias` embedding. We preserve it:

```python
if hasattr(src, "relative_attention_bias"):
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

## 5.3 Nested Encoder Handling

For RoBERTa/BERT, we use `IdentityOutProj` for the inner attention's output 
projection:

```python
class IdentityOutProj(nn.Identity):
    lora_A = None
    lora_B = None
    def freeze_base_weights(self): pass
```

The outer `attention.output.dense` (in `BertAttention`) applies the actual 
projection.

## 5.4 Causality

For decoders, we apply the causal mask **before** the ramp:

```python
causal_tril = torch.tril(torch.ones((L, L), dtype=torch.bool))
scores = scores.masked_fill(~causal_tril, -10000.0)
# Then apply ramp on distances
```

For causal distance, $d_{ij} = \max(0, i - j)$.

## 5.5 RoPE Handling

RoPE is applied to Q, K **before** computing scores:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
```

Then standard score computation, mask, ramp.

## 5.6 Fast Path

If all heads have `span + R >= L`, we skip ramp computation entirely. This 
avoids allocating $\mathcal{O}(H L^2)$ when unnecessary.

```python
span = torch.sigmoid(self.span_logits) * self.max_span
if not ((span + self.ramp_size) >= N_k).all():
    # allocate ramp
    ...
```

## 5.7 Injection Algorithm

**Algorithm 2:** Inject ASA-Span into a full model
```
Input:  model M, config c, S_max, R
Output: M with ASA-Span attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      if is_already_adaptive_span(src): continue
5:      new ← AdaptiveSpanAttention(config=c, pretrained_attn=src,
6:                                    max_span=S_max, ramp_size=R)
7:      replace_attention(ℓ, new)
8: end for
```

## 5.8 Failure Handling

`inject_adaptive_span` returns `None` on:
- Already-injected layer (AdaptiveSpan instance)
- DeltaNet/LinearAttention layers (unsupported)
- Weight extraction failure

```python
try:
    new_block = AdaptiveSpanAttention(...)
except Exception:
    return None
```

## 5.9 Parameter Count

| Projection | Params |
|---|---|
| $W_Q$ | $d \cdot H d_h$ |
| $W_K$ | $d \cdot H_{kv} d_h$ |
| $W_V$ | $d \cdot H_{kv} d_h$ |
| $W_O$ | $H d_h \cdot d$ |
| Span logits | $H$ |

Total: same as MHA + $H$ extra parameters.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_span")
@ATTENTION_REGISTRY.register("adaptive_span_attention")
class AdaptiveSpanAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| Already injected | Type check | Skip layer |
| DeltaNet layer | Class name check | Skip layer |
| RoPE unavailable | Exception | Fallback |
| CUDA OOM | RuntimeError | Reduce batch |

---
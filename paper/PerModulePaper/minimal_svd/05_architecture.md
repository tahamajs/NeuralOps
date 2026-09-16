<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Nested? | Conv1D? |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are `nn.Linear`. Direct factorization, no special handling.

`is_nested_encoder = False` because `out_lin` is inside the attention.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query  (Linear 768 → 768)
│   ├── key    (Linear 768 → 768)
│   └── value  (Linear 768 → 768)
└── output
    └── dense  (Linear 768 → 768)   # ← Output projection OUTSIDE
```

**Critical:** `output.dense` is **outside** `self`. When MinimalSVD replaces 
`self`, the outer `output.dense` must still apply. We set 
`is_nested_encoder = True` and skip `out_proj` in the forward pass.

**Detection.** MinimalSVD sets `is_nested_encoder` in two places:

1. In `__init__`, based on `model_type in ("roberta", "bert", "xlm-roberta")`.
2. In `copy_weights`, based on the presence of `src.self.query`.

If they disagree, `copy_weights` wins.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)   # ← FUSED Q, K, V
└── c_proj  (Conv1D 768 → 768)    # ← output projection
```

**Two challenges:**

1. **Fused c_attn.** Q, K, V are stored in one matrix. We split:
   ```python
   w = src.c_attn.weight.data.clone()
   if is_conv1d:
       w = w.t()   # (in, out) → (out, in)
   q_w, k_w, v_w = w.chunk(3, dim=0)
   ```

2. **Conv1D orientation.** HuggingFace `Conv1D` stores weights as `(in, out)`, 
   opposite to `nn.Linear`. Detected via:
   ```python
   is_conv1d = hasattr(module, "nf") or "Conv1D" in type(module).__name__
   ```

`is_nested_encoder = False` because `c_proj` is inside the attention (applied 
by our module).

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA handling:** $H_{kv} = 2 < H = 8$. K and V projections are narrower. 
MinimalSVD factorizes each at its natural dimension. During forward:

```python
if self.num_kv_heads < self.num_heads:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

**RoPE handling:** Qwen applies RoPE after projection.

`is_nested_encoder = False` because `o_proj` is inside the attention.

## 5.6 Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
def forward(self, ...):
    ...
    context = torch.matmul(attn_weights, V)
    context = context.transpose(1, 2).contiguous().view(B, N, H * D)
    
    if self.is_nested_encoder:
        return context, attn_weights   # Raw context
    else:
        output = self.out_proj(context)
        return output, attn_weights
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject MinimalSVD (single-layer)
```
Input:  layer, config, energy η
Output: layer with factorized attention

1:  src ← get_attention(layer)
2:  if src is MinimalSVD: return None
3:
4:  new_attn ← MinimalSVDAttention(config)
5:  try:
6:      new_attn.copy_weights(src)
7:  except Exception as e:
8:      return None
9:
10: p ← next(src.parameters())
11: new_attn.to(device=p.device, dtype=p.dtype)
12:
13: # Set nested flag and replace
14: if hasattr(layer, "attention"):
15:     if hasattr(layer.attention, "self"):
16:         layer.attention.self = new_attn
17:         new_attn.is_nested_encoder = True
18:     else:
19:         layer.attention = new_attn
20:         new_attn.is_nested_encoder = False
21: elif hasattr(layer, "self"):
22:     layer.self = new_attn
23:     new_attn.is_nested_encoder = True
24: return new_attn
```

**Whole-model traversal** is the caller's responsibility:

```python
for layer in model.distilbert.transformer.layer:
    inject_minimal_svd(layer, model.config, energy=0.95)
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | Padding in `decompose_from_weight` | Warn |
| Cross-attention | Not supported | Not implemented |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("minimal_svd")
class MinimalSVDAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("minimal_svd_inject")
@ATTENTION_REGISTRY.register("inject_minimal_svd")
def inject_minimal_svd(...):
    ...
```

## 5.10 Interaction with LoRA Recovery Pipeline

MinimalSVD integrates cleanly with the framework's LoRA recovery:

1. **Inject** MinimalSVD → factorized attention with frozen base.
2. **LoRA adapters** are inside each `FactorisedLinear` (trainable).
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

## 5.11 Relationship to Companion Variants

| Feature | MinimalSVD | NOVA | PRISM |
|---|---|---|---|
| Whole-model traversal | ❌ | ✅ | ✅ |
| Pre-test on first layer | ❌ | ✅ | ✅ |
| Injection-time nested detection | ❌ | ❌ | ✅ |
| Double-injection guard | ❌ | ❌ | ✅ |
| Code lines | ~280 | ~500 | ~550 |

---
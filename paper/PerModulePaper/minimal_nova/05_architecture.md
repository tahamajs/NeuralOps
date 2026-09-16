<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Attention | Nested? | Conv1D? |
|---|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | Flat | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Nested | Yes | No |
| GPT-2 | 12 | 768 | 12 | 12 | Causal | No | Yes |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | GQA+RoPE | No | No |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are `nn.Linear`. **Direct factorization**, no special handling.

`is_nested_encoder = False` because `out_lin` is inside the attention 
module and must be applied by MinimalNOVA.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query  (Linear 768 → 768)
│   ├── key    (Linear 768 → 768)
│   └── value  (Linear 768 → 768)
└── output
    └── dense  (Linear 768 → 768)   # ← output projection OUTSIDE
```

**Critical:** `output.dense` sits **outside** `self`. When MinimalNOVA 
replaces `self`, the outer `output.dense` must still apply.

`is_nested_encoder = True` because:
- MinimalNOVA returns the attention **context** (before output projection).
- The outer `output.dense` applies the projection.

If we incorrectly set `is_nested_encoder = False`, `output.dense` is applied 
**twice**, causing scale collapse.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)   # ← FUSED Q, K, V
└── c_proj  (Conv1D 768 → 768)    # ← output projection
```

**Two challenges:**

1. **Fused c_attn.** Q, K, V are stored in one matrix. We split it:
   ```python
   w = src.c_attn.weight.data.clone()
   if "Conv1D" in type(src.c_attn).__name__:
       w = w.t()   # (in, out) → (out, in)
   q_w, k_w, v_w = w.chunk(3, dim=0)
   ```

2. **Conv1D orientation.** HuggingFace's `Conv1D` stores weights as 
   `(in, out)`, opposite to `nn.Linear`. We detect this via:
   ```python
   is_conv1d = hasattr(module, "nf") or "Conv1D" in type(module).__name__
   if is_conv1d:
       w = w.t()
   ```

`is_nested_encoder = False` because `c_proj` is applied by our module.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h = 8 * 256
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h = 2 * 256
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA handling:** $H_{kv} = 2 < H = 8$. K and V projections are narrower. 
MinimalNOVA factorizes each at its natural dimension. During forward:

```python
if self.num_kv_heads < self.num_heads:
    K = repeat_kv(K, self.kv_repeat)   # (B, 2, N, 256) → (B, 8, N, 256)
    V = repeat_kv(V, self.kv_repeat)
```

**RoPE handling:** Qwen applies RoPE after projection. MinimalNOVA preserves 
this via `position_embeddings` argument:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

`is_nested_encoder = False` because `o_proj` is inside the attention.

## 5.6 The Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
def forward(self, ...):
    ...
    context = torch.matmul(attn_weights, V)
    context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, attn_weights   # ← Raw context, no out_proj
    else:
        output = self.out_proj(context)
        return output, attn_weights
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject MinimalNOVA into a full model
```
Input:  model M, rank_cap r_max, energy η
Output: M' with all attention blocks factorized

1:  layer_list ← find_transformer_layers(M)
2:  # Pre-test on first layer
3:  test_src ← get_attention(layer_list[0])
4:  test_module ← MinimalNOVA(cfg)
5:  try: test_module.copy_weights(test_src)
6:  except: return None   # abort
7:
8:  # Inject each layer
9:  count ← 0
10: for (idx, layer) in enumerate(layer_list) do
11:     res ← inject_single(layer, idx, r_max, η)
12:     if res is not None:
13:         count += 1
14: end for
15:
16: return M' if count > 0 else None
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V weights | `copy_weights` raises | Abort |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | `_orient` handles | Warn and pad |
| Double injection | Class name check | Skip |
| Cross-attention | Not supported | Raise |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("minimal_nova")
class MinimalNOVAAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("minimal_nova_inject")
@ATTENTION_REGISTRY.register("inject_minimal_nova")
def inject_minimal_nova(...):
    ...
```

## 5.10 Compatibility with LoRA Recovery Pipeline

MinimalNOVA integrates cleanly with the framework's LoRA recovery:

1. **Inject** MinimalNOVA → factorized attention with frozen base.
2. **LoRA adapters** are inside each `FactorisedLinear` (trainable).
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

The framework's `get_recovery_params` finds `lora_A`, `lora_B` parameters 
automatically via the naming convention.

---
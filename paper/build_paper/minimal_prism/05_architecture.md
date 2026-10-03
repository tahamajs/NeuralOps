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

**Critical:** `output.dense` sits **outside** `self`. When MinimalPRISM 
replaces `self`, the outer `output.dense` must still apply.

`is_nested_encoder = True` because MinimalPRISM returns the raw context 
(before output projection), and the outer `output.dense` applies the 
projection.

**Failure mode.** If we incorrectly set `is_nested_encoder = False`, 
`output.dense` is applied **twice**, causing scale collapse. MinimalPRISM 
avoids this via injection-time detection.

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
       w = w.t()  # (in, out) → (out, in)
   q_w, k_w, v_w = w.chunk(3, dim=0)
   ```

2. **Conv1D orientation.** HuggingFace `Conv1D` stores weights as 
   `(in_features, out_features)`, opposite to `nn.Linear`. Detected via:
   ```python
   is_conv1d = hasattr(module, "nf") or "Conv1D" in type(module).__name__
   ```

`is_nested_encoder = False` because `c_proj` is applied by our module.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA handling:** $H_{kv} = 2 < H = 8$. K and V projections are narrower. 
MinimalPRISM factorizes each at its natural dimension. During forward:

```python
if self.num_kv_heads < self.num_heads:
    K = repeat_kv(K, self.kv_repeat)  # (B, 2, N, 256) → (B, 8, N, 256)
    V = repeat_kv(V, self.kv_repeat)
```

**RoPE handling:** Qwen applies RoPE after projection. MinimalPRISM preserves 
this via `position_embeddings` argument.

`is_nested_encoder = False` because `o_proj` is inside the attention.

## 5.6 Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
def forward(self, ...):
    ...
    context = torch.matmul(attn_weights, V)
    context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, attn_weights   # Raw context
    else:
        output = self.out_proj(context)
        return output, attn_weights
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject MinimalPRISM into a full model
```
Input:  model M, rank_cap r_max, energy η
Output: M' with all attention blocks factorized

1:  layer_list ← find_transformer_layers(M)
2:
3:  # Pre-test
4:  test_src ← get_attention(layer_list[0])
5:  test_module ← MinimalPRISMAttention(cfg)
6:  try: test_module.copy_weights(test_src)
7:  except: return None
8:
9:  # Inject each layer
10: count ← 0
11: for (idx, layer) in enumerate(layer_list) do
12:     src_attn ← get_attention(layer)
13:     if src_attn is already MinimalPRISM: skip
14:     new_attn ← MinimalPRISMAttention(cfg)
15:     new_attn.copy_weights(src_attn)
16:
17:     # Set nested flag at injection
18:     if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
19:         new_attn.is_nested_encoder = True
20:         layer.attention.self = new_attn
21:     else:
22:         new_attn.is_nested_encoder = False
23:         replace_attention(layer, new_attn)
24:     count += 1
25: end for
26:
27: return M' if count > 0 else None
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Abort (pre-test) |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | `_orient` and padding | Warn |
| Double injection | Class name check | Skip |
| Cross-attention | Not supported | Not implemented |
| CUDA OOM during SVD | RuntimeError | Fallback to CPU |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("minimal_prism")
class MinimalPRISMAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("minimal_prism_inject")
@ATTENTION_REGISTRY.register("inject_minimal_prism")
def inject_minimal_prism(...):
    ...
```

## 5.10 Interaction with LoRA Recovery Pipeline

MinimalPRISM integrates cleanly with the framework's LoRA recovery:

1. **Inject** MinimalPRISM → factorized attention with frozen base.
2. **LoRA adapters** are inside each `FactorisedLinear` (trainable).
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

The framework's `get_recovery_params` finds `lora_A`, `lora_B` parameters 
automatically via the naming convention.

---
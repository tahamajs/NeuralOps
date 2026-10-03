<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | $N$ tested | Nyström path | Fallback |
|---|---|---|---|
| DistilBERT | 128 | ❌ | Dense |
| RoBERTa | 128 | ❌ | Dense |
| GPT-2 | 128 | ❌ (causal) | Dense causal |
| Qwen3.5 | 128 | ❌ (causal) | Dense causal |

**Note.** At $N = 128$, all four models use the **dense fallback**. The 
Nyström branch only activates for $N > 128$.

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin
├── k_lin
├── v_lin
└── out_lin
```

**Extraction:**
```python
w_q = src.q_lin.weight.data
w_k = src.k_lin.weight.data
w_v = src.v_lin.weight.data
w_o = src.out_lin.weight.data
```

`is_nested_encoder = False`.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query, key, value
└── output
    └── dense
```

**Nested detection:**
```python
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested = True
```

`is_nested_encoder = True`. The outer `output.dense` applies projection.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D)
└── c_proj  (Conv1D)
```

**Split c_attn:**
```python
c_w = src.c_attn.weight.data
if "Conv1D" in type(src.c_attn).__name__ or c_w.shape[0] < c_w.shape[1]:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

**Causal decoder detection:** `self.is_decoder = True` → dense causal 
attention.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj
├── k_proj
├── v_proj
└── o_proj
```

**GQA:** `repeat_kv` during forward.

**RoPE:** applied after projection via `position_embeddings`.

**Causal decoder detection:** `self.is_decoder = True`.

## 5.6 Nested Encoder Bypass

```python
context = torch.matmul(attn_weights, V)
context = context.transpose(1, 2).contiguous().view(...)

if self.is_nested_encoder:
    return context, None    # Skip out_proj
else:
    return self.out_proj(context), None
```

## 5.7 Injection Hook

```python
def inject_nystrom(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or layer.attention...
    
    # Extract dimensions
    hidden_size = getattr(src_attn, 'hidden_size', ...)
    num_heads = getattr(src_attn, 'num_heads', ...)
    
    try:
        landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config else 32
        new_attn = NystromAttention(
            config=model_config,
            num_landmarks=landmarks,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        
        # Replace
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"injection failed: {e}")
        return None
```

**Safe injection:** returns `None` on failure. Caller can continue with 
original attention.

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Abort |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Cross-attention | Not supported | Raise |
| Short sequence | $N \leq 128$ | Dense |
| Causal | `is_decoder = True` | Dense causal |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("nystrom")
@ATTENTION_REGISTRY.register("nystrom_attention")
class NystromAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("nystrom_inject")
@ATTENTION_REGISTRY.register("inject_nystrom")
def inject_nystrom(...):
    ...
```

## 5.10 LoRA Recovery Integration

Nyströmformer uses LoRA adapters on all four projections:

- `query.lora_A`, `query.lora_B`
- `key.lora_A`, `key.lora_B`
- `value.lora_A`, `value.lora_B`
- `out_proj.lora_A`, `out_proj.lora_B`

Base weights ($W_Q, W_K, W_V, W_O$) are frozen. Only LoRA adapters trained.

## 5.11 Comparison with Original

| Aspect | Original (2021) | Our reproduction |
|---|---|---|
| Target | BERT | Four families |
| Landmarks | Adaptive pool | Same |
| Pinv | Newton-Schulz | Direct pinv + NS fallback |
| Causal | Not addressed | Explicit dense fallback |
| Short seq | Not specified | Fallback at $N \leq 128$ |
| Multi-arch | No | Yes |

---
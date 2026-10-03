<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Nested? | Conv1D? | RoPE? |
|---|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No | **Added** |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No | **Added** |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes | **Added** |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No | Native |

**Note.** DistilBERT, RoBERTa, and GPT-2 use absolute position embeddings. 
UnifiedRoPE **adds** RoPE on top, which may not match pretrained behavior.

For Qwen, RoPE is native — UnifiedRoPE simply preserves it.

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

**Weight extraction:**
```python
w_q = src.q_lin.weight.detach()
w_k = src.k_lin.weight.detach()
w_v = src.v_lin.weight.detach()
w_o = src.out_lin.weight.detach()
```

`is_nested_encoder = False`.

**RoPE caveat.** DistilBERT was trained with absolute positions. Adding 
RoPE changes the attention pattern. Accuracy may drop unless recovery 
fine-tuning adapts.

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
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    src = old_attn.self
    w_o = old_attn.output.dense.weight
```

`is_nested_encoder = True`. Forward returns raw context.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)
└── c_proj  (Conv1D 768 → 768)
```

**Split c_attn:**
```python
c_attn_w = src.c_attn.weight.detach()
if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
    c_attn_w = c_attn_w.t()
q_dim = self.q_head_size
k_dim = self.kv_head_size
v_dim = self.kv_head_size
w_q = c_attn_w[:q_dim, :]
w_k = c_attn_w[q_dim : q_dim + k_dim, :]
w_v = c_attn_w[q_dim + k_dim :, :]
```

`is_nested_encoder = False`. Causal mask applied in forward.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)
├── k_proj  (Linear 1024 → 512)
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA expansion:**
```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)
```

**RoPE:** Either internal `_apply_rope` or external `position_embeddings`.

`is_nested_encoder = False`.

## 5.6 Injection Hook

```python
def inject_unified_rope(layer, model_config=None, comp_config=None, old_attn=None, **kwargs):
    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                src = getattr(layer, attr)
                break

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config

    lora_rank = kwargs.pop("lora_rank", 16)
    rank = kwargs.pop("rank", 32)

    new_attn = UnifiedRoPE(
        config=model_config,
        rank=rank,
        lora_rank=lora_rank,
        pretrained_attn=src,
        **kwargs,
    )

    # Replace
    for attr in ("self_attn", "attn", "self"):
        if hasattr(layer, attr):
            setattr(layer, attr, new_attn)
            return new_attn
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
        return new_attn
    layer.attention = new_attn
    return new_attn
```

**Safe injection.** Wraps in try/except. Logs errors. **Raises** 
`RuntimeError` on failure (does not silently return None).

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Validation | Raise RuntimeError |
| Near-zero Frobenius | Norm check < 1e-4 | Raise RuntimeError |
| Cross-attention | Not supported | Not implemented |
| Conv1D wrong orientation | Shape check | Transpose |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("compression_unified_rope")
@ATTENTION_REGISTRY.register("unified_rope")
@ATTENTION_REGISTRY.register("unified_rope_attention")
class UnifiedRoPE(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("unified_rope_inject")
@ATTENTION_REGISTRY.register("compression_unified_rope_inject")
@ATTENTION_REGISTRY.register("inject_unified_rope")
def inject_unified_rope(...):
    ...
```

## 5.9 LoRA Recovery Integration

UnifiedRoPE uses `LoRALinear` projections with `freeze_base=True`. Base 
weights are frozen; only LoRA adapters trained during recovery.

## 5.10 Comparison with Companion Methods

| Method | Type | Purpose |
|---|---|---|
| MinimalSVD | Compression | Reduce params |
| Dobi-SVD | Compression | Reduce params |
| MLA | Compression | Reduce KV |
| Nyströmformer | Compute | Reduce compute |
| **UnifiedRoPE** | **Replacement** | **Portability** |

**UnifiedRoPE is orthogonal** to all compression methods.

---
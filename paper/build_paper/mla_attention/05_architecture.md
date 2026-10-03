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
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No | No |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes | No |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No | Yes |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
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
│   ├── query
│   ├── key
│   └── value
└── output
    └── dense
```

**Extraction:**
```python
src_inner = src.self
w_q = src_inner.query.weight.data
w_k = src_inner.key.weight.data
w_v = src_inner.value.weight.data
w_o = src.output.dense.weight.data
```

**Nested detection:** `hasattr(src, "self") and hasattr(src.self, "query")`.

`is_nested_encoder = True`. Forward returns raw context; outer 
`output.dense` applies projection.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)
└── c_proj  (Conv1D 768 → 768)
```

**Extraction:**
```python
weight = src.c_attn.weight.data
if weight.shape[1] == 3 * hidden_size:   # Conv1D stores (in, out)
    weight = weight.t()
q_w, k_w, v_w = weight.chunk(3, dim=0)
w_o = src.c_proj.weight.data
```

`is_nested_encoder = False`.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)
├── k_proj  (Linear 1024 → 512)
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA dimension match:**
```python
if w_k_f.shape[0] > self.kv_dim:
    rep = w_k_f.shape[0] // self.kv_dim
    w_k_f = w_k_f.view(self.kv_dim, rep, -1).mean(dim=1)
```

**RoPE applied after up-projection** via `position_embeddings` in kwargs.

`is_nested_encoder = False`.

## 5.6 Injection Hook

```python
def inject_mla(layer, model_config, comp_config=None, old_attn=None, ...):
    rank = getattr(comp_config, "mla_rank", 128) if comp_config else 128
    new_attn = MLAAttention(model_config, rank=rank)
    
    src_attn = old_attn or getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        new_attn.copy_weights(src_attn)
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
    
    return new_attn
```

**Note.** This injection hook is **single-layer only**. Whole-model traversal 
is the caller's responsibility.

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `copy_weights` raises | Abort |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Wrong Conv1D | `hasattr(src, "c_attn")` | Handle |
| Missing out_proj (nested) | `is_nested_encoder` | Identity matrix |
| GQA dimension mismatch | Shape check | Mean-pool |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("mla")
@ATTENTION_REGISTRY.register("mla_attention")
class MLAAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("mla_inject")
def inject_mla(...):
    ...
```

## 5.9 LoRA Recovery Integration

MLA integrates with the framework's LoRA recovery:

1. MLA initialization → frozen base, trainable LoRA on Q/O.
2. Recovery fine-tuning trains LoRA + `output_scale`.
3. Evaluation uses the fine-tuned model.

## 5.10 Cross-Attention

MLA does **not** support cross-attention. Encoder-decoder models (T5, BART) 
are not applicable.

## 5.11 Comparison with Original DeepSeek-V2 MLA

| Aspect | Original | Our reproduction |
|---|---|---|
| Target architecture | DeepSeek-V2 | Four families |
| Latent dim | 512 (V2), 1536 (V3) | 32 |
| RoPE on K | Decomposed | After projection |
| GQA | Native | Supported via pool |
| Output scale | Not specified | Trainable scalar |

**Differences from original.**

1. **Latent dimension.** We use 32 instead of 512.
2. **RoPE handling.** The original applies RoPE on a decoupled head 
   dimension; we apply it uniformly after up-projection.
3. **Multi-architecture support.** The original does not target DistilBERT, 
   RoBERTa, or GPT-2.

---
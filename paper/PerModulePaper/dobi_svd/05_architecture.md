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

## 5.2 Target Module Selection

Dobi-SVD with `target_attention_only = True` (default) targets only 
attention projections:

| Pattern | Examples |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Qwen |
| `q_lin`, `k_lin`, `v_lin`, `out_lin` | DistilBERT |
| `c_attn`, `c_proj` | GPT-2 |
| `query`, `key`, `value` | RoBERTa |
| Anything containing `attn` or `attention` | Any |

With `target_attention_only = False`, all compressible layers are targeted 
(including FFN). Non-target layers are frozen.

## 5.3 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are targeted. Direct SVD, no special handling.

## 5.4 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query, key, value
└── output
    └── dense
```

The `query, key, value` are targeted. `output.dense` may be targeted if 
`target_attention_only = False`.

Nested encoder detection is **not** required because Dobi-SVD operates on 
**individual linear layers**, not on the attention module as a whole. It 
simply replaces `query` → `DobiSVDLinearLayer(query)`.

## 5.5 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)
└── c_proj  (Conv1D 768 → 768)
```

**Conv1D handling:** Dobi-SVD detects Conv1D by checking:

```python
is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
```

and transposes weights before SVD. The output shape is preserved 
(Conv1D stores `(in, out)`; Dobi-SVD expects `(out, in)`).

**Attention-only filtering:** `c_attn` and `c_proj` are in the `_ATTN_NAMES` 
set, so they are targeted.

## 5.6 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)
├── k_proj  (Linear 1024 → 512)
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

Each is targeted individually. GQA is handled at the **attention level**, 
not by Dobi-SVD. The compressor just factorizes the linear layers.

## 5.7 LoRALinear Wrapping

When a target layer is inside a `LoRALinear` wrapper, Dobi-SVD preserves 
the wrapper:

```python
if hasattr(module, "base_layer"):
    module.base_layer = new_layer
elif hasattr(module, "linear") and hasattr(module, "lora_A"):
    module.linear = new_layer
else:
    replace_module(self.model, name, new_layer)
```

After replacement, LoRA parameters remain trainable:

```python
for p_name, p in module.named_parameters():
    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
        p.requires_grad = True
```

## 5.8 Injection Hook

Dobi-SVD uses a **compressor-level API** (not an attention-level one):

```python
def dobi_factory(model, comp_config, **kwargs):
    compressor = DobiSVDCompressor(model, ...)
    compressed_model, stats = compressor.compress()
    return compressed_model

@COMPRESSOR_REGISTRY.register("inject_dobi")
def inject_dobi(model=None, layer=None, **kwargs):
    target = layer if layer is not None else model
    try:
        return dobi_factory(target, **kwargs)
    except Exception as e:
        logger.error("inject_dobi failed: %s", e)
        return None
```

**Note.** Dobi-SVD works on the **whole model** or on a **single layer**. 
For whole-model, it traverses `named_modules()`.

## 5.9 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing bias info | Shape check | Infer from weight |
| Conv1D orientation | `hasattr(target, "nf")` | Transpose |
| Reconstruction error high | Frobenius check | Retry, then fallback |
| Near-zero Frobenius | Norm check | Skip layer |
| SVD failure | Try/except | Skip layer |
| LoRA wrapper | `hasattr(module, "base_layer")` | Preserve wrapper |

## 5.10 Registry

```python
@COMPRESSOR_REGISTRY.register("dobi")
@COMPRESSOR_REGISTRY.register("dobi_svd")
@COMPRESSOR_REGISTRY.register("dobi_compressor")
@COMPRESSOR_REGISTRY.register("dobi_svd_compressor")
@COMPRESSOR_REGISTRY.register("dobi_factory")
@COMPRESSOR_REGISTRY.register("dobi_svd_factory")
def dobi_factory(model, comp_config, **kwargs):
    ...
```

## 5.11 Interaction with LoRA Recovery Pipeline

Dobi-SVD integrates cleanly with the framework's LoRA recovery:

1. **Compress** target layers → `DobiSVDLinearLayer` with gates.
2. **Gates** are trainable, LoRA adapters (if wrapped) are trainable.
3. **Recovery fine-tuning** trains gates + LoRA.
4. **Post-hoc pruning** (optional) removes near-zero gates.
5. **Evaluation** uses the compressed model.

---
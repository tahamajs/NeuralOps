<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Resolution

EAT resolves `hidden_size` and `num_heads` in this order:

1. `config.hidden_size` / `config.num_attention_heads`
2. From `source_attn` projections (`q_lin.in_features`, etc.)
3. From `kwargs` (`hidden_size`, `num_heads`)
4. Default `768` / `12`

This ensures EAT works even when `config` is incomplete.

## 5.3 Layer Targeting

The injection hook searches for attention modules in this order:

1. `layer.attention` (RoBERTa/BERT)
2. `layer.self_attn` (Qwen/LLaMA)
3. `layer.attn` (GPT-2)
4. `layer.self` (DistilBERT variants)

For nested encoders, `layer.attention.self` is targeted.

## 5.4 Copy Weights — Case by Case

### 5.4.1 DistilBERT

```python
# Flat attention — all four projections available
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### 5.4.2 RoBERTa

```python
# Nested — extract from src.self, skip out_proj
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

The `is_nested_encoder = True` flag means EAT returns the raw context 
without applying `out_lin`. The outer `attention.output.dense` applies it.

### 5.4.3 GPT-2

```python
# Fused Conv1D — transpose and split
w = src.c_attn.weight.data.t()  # (3d, d)
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:d])
extracted["k"] = (k_w, src.c_attn.bias.data[d : 2 * d])
extracted["v"] = (v_w, src.c_attn.bias.data[2 * d :])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.4.4 Qwen (GQA + RoPE)

```python
# Direct copy from named projections
extracted["q"] = (src.q_proj.weight.data, src.q_proj.bias.data)
extracted["k"] = (src.k_proj.weight.data, src.k_proj.bias.data)
extracted["v"] = (src.v_proj.weight.data, src.v_proj.bias.data)
extracted["o"] = (src.o_proj.weight.data, src.o_proj.bias.data)
# RoPE preserved via rotary_emb copy
```

## 5.5 The `is_nested_encoder` Flag

This flag determines whether the module applies its own `out_lin` projection:

- **True** (RoBERTa): output projection is applied externally; EAT returns 
  raw context.
- **False** (DistilBERT, GPT-2, Qwen): EAT applies `out_lin` internally.

**Common bug.** If `is_nested_encoder` is set incorrectly, the projection 
is applied twice (resulting in a scale collapse) or zero times (resulting 
in missing output).

The detection logic:

```python
if (
    hasattr(src, "self")
    and hasattr(src.self, "query")
    and hasattr(src, "output")
    and hasattr(src.output, "dense")
):
    is_nested = True
elif cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower() and not has_out_lin:
    is_nested = True
```

## 5.6 Injection

```python
def inject_eat(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = EATBlock(config=model_config, pretrained_attn=src, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
```

The replacement targets the same attribute path as the source:

```python
for attr in ("attention", "self_attn", "attn", "self"):
    if hasattr(layer, attr):
        setattr(layer, attr, new_attn)
        break
```

## 5.7 Handling LoRA-Wrapped Source Modules

If the source module is already wrapped with LoRA:

1. **Merge LoRA** into the base layer (`c_attn_mod.merge()`).
2. **Extract** merged weights.
3. **Re-initialize** LoRA in EAT with the same rank.

This ensures the weights transferred are the effective weights, not the 
base weights.

## 5.8 Failure Modes and Silent Fallback

EAT's `copy_weights` is defensive:

| Failure | Action |
|---|---|
| LoRA merge fails | Log debug, continue with unmerged weights |
| Missing projection | Log warning, keep random init for that projection |
| Small Frobenius norm | Log warning, continue |
| UniversalAdapter fails | Log debug, continue |

**The critical consequence:** If weight transfer fails silently, EAT 
initializes with random weights, and the model's forward pass produces 
garbage. Recovery fine-tuning cannot compensate because gradient signals 
are dominated by noise.

**Mitigation.** We recommend:

1. **Add a hard assertion** (optional flag `strict_weight_transfer=True`) 
   that raises if any projection is missing.
2. **Post-injection sanity check:** Compare logits before and after 
   injection on a fixed input. If the difference is large, abort.
3. **Log at WARN level** on any missing projection (currently does).

## 5.9 The `EATBlock` Module

```python
class EATBlock(BaseAttention):
    def __init__(self, config, rank=64, ..., prune_layer=None):
        super().__init__(config, rank)
        self.q_lin = LoRALinear(d, H*d_h, lora_rank=16)
        self.k_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.v_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_lin = LoRALinear(H*d_h, d, lora_rank=16)
        self.prune_layer = prune_layer
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
    
    def forward(self, X, attention_mask=None, ...):
        Q = self.q_lin(X).view(...)
        K = self.k_lin(X).view(...)
        V = self.v_lin(X).view(...)
        # attention, output projection, optional pruning
        return Y
```

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("eat")
@ATTENTION_REGISTRY.register("eat_attention")
class EATBlock(BaseAttention): ...
```

## 5.11 Causality Verification

For decoder models, EAT maintains causality because:

1. **Attention masking:** Causal mask is applied in `scores`.
2. **No future leakage in projections:** Q/K/V projections are 
   position-wise.
3. **Pruning causality:** Progressive pruning, if used, must respect 
   causal ordering (see `ProgressiveTokenPruning` documentation).

**Verification.**

```python
X = torch.randn(1, L, d, requires_grad=True)
Y = model(X)
loss = Y[0, i].sum()
loss.backward()
assert X.grad[0, i + 1 :].abs().max() < 1e-6
```

32

---
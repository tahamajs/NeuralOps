# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Extraction

MinimalAURORA uses a **layered extraction strategy**:

### 5.2.1 Priority Order

1. **UniversalAdapter** — tried first, covers many cases.
2. **GPT-2 fused `c_attn`** — transpose and split.
3. **RoBERTa nested `self.query/key/value`** — extract from inner block.
4. **Named mappings** — `q_proj`, `q_lin`, `query`, etc.

### 5.2.2 GPT-2 Conv1D Handling

```python
if ("Conv1D" in p_cls or hasattr(base_c, "nf")
    or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size)):
    weight = weight.t()
```

The heuristic `weight.shape[1] == 3 * hidden` catches the case where 
Conv1D stores weights as `(in, 3*out)`.

### 5.2.3 RoBERTa Nested Detection

Detection logic:

```python
if hasattr(src, "self") and hasattr(src.self, "query"):
    is_nested = True
elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention", ...)
      or (cls_name.endswith("SelfAttention") and not has_out_proj)):
    is_nested = True
self.is_nested_encoder = is_nested
```

For DistilBERT: `MultiHeadSelfAttention` ends with "SelfAttention" but 
has `out_lin` → NOT nested. Good.

For RoBERTa: `BertAttention` has `self` and `output` → nested. Good.

### 5.2.4 Nested Encoder Output Projection

For nested encoders, the output projection is set to **identity**:

```python
elif proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

And in forward:

```python
if self.is_nested_encoder:
    output = context * self.output_scale
else:
    output = self.out_proj(context) * self.output_scale
```

## 5.3 Injection

```python
def inject_minimal_aurora(layer, model_config, old_attn=None, **kwargs):
    new_attn = MinimalAURORAAttention(model_config=model_config, ...)
    
    src_attn = old_attn or find_attention(layer)
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error(f"copy_weights failed: {e}")
            return None  # silent failure
        # transfer device/dtype
        p = next(src_attn.parameters())
        new_attn.to(device=p.device, dtype=p.dtype)
    
    # replace attention
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    
    return new_attn
```

**Note.** The injection returns `None` on failure, which is **silent**. 
Unlike A3/AASVD (which raise), this requires the caller to check the 
return value.

## 5.4 The `FactorisedLinear` Module

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, bias=True,
                 energy=0.95, lora_rank=16, lora_alpha=32.0):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
```

## 5.5 The `output_scale` Parameter

```python
self.output_scale = nn.Parameter(torch.ones(1))
```

Applied in forward:

```python
output = self.out_proj(context) * self.output_scale
```

This is a new parameter not present in the base checkpoint. It 
initializes to 1.0 and is trainable. During recovery training, it 
may drift. If it reaches 0, the entire attention output collapses.

## 5.6 The `pres_gate` Parameter (Unused)

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

**This parameter is defined but never used in the forward pass.** It 
appears to be a leftover from a previous design (perhaps a preservation 
gate between attention output and residual). We document this in §8.6 
and recommend removal.

## 5.7 Failure Modes

| Failure | Response |
|---|---|
| Missing projection | `copy_weights` raises RuntimeError |
| Small Frobenius norm | Raises RuntimeError |
| Injection error | Returns None (silent) |
| `output_scale` drift to 0 | Attention output collapses |
| `pres_gate` unused | No effect |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("minimal_aurora", supports_causal=True, supports_gqa=True)
class MinimalAURORAAttention(BaseAttention):
    ...
```

## 5.9 Cross-Attention

Forward accepts `encoder_hidden_states` and supports cross-attention 
(though GQA + cross-attention interaction may have edge cases).

---
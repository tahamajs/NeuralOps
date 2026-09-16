<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention class | Nested? | use_full_attention=True | use_full_attention=False |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | Fragile transfer | GatedDeltaNet (slow) |
| RoBERTa | `BertSelfAttention` | Yes | Fragile transfer | GatedDeltaNet (slow) |
| GPT-2 | `GPT2Attention` | No | Fragile transfer | GatedDeltaNet (slow) |

## 5.2 Injection Algorithm

**Algorithm 2:** Inject GatedDeltaNetBlock
```
Input:  layer ℓ, config, old_attn
Output: new block in ℓ

1: Determine d_model, num_heads from config or src
2: new ← GatedDeltaNetBlock(config, d_model, num_heads, use_full_attention)
3: if src_attn is not None:
4:     new.copy_weights(src_attn)      ▷ May fail silently
5: Replace attention slot:
6:     if ℓ has .attention.self: ℓ.attention.self ← new; is_nested ← True
7:     elif ℓ has .attention: ℓ.attention ← new
8:     elif ℓ has .self: ℓ.self ← new; is_nested ← True
9:     elif ℓ has .attn: ℓ.attn ← new
10: return new
```

## 5.3 The Nested Encoder Path

For RoBERTa, `is_nested_encoder=True`. The block returns **raw attention 
output** without residual or LayerNorm:

```python
if self.is_nested_encoder:
    return delta_out, None
```

**Why?** Because in RoBERTa, the outer `BertAttention` block already wraps 
attention in a residual + LayerNorm:

```python
class BertAttention(nn.Module):
    def __init__(...):
        self.self = BertSelfAttention(...)
        self.output = BertSelfOutput(...)  # has dense + LayerNorm + residual
    
    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,)
```

So replacing `layer.attention.self` means the outer `output` block handles 
residual + LayerNorm. Our block must return the **raw context**.

**However.** Our block's `self.norm` is **not used** in the nested path. So 
the `self.norm` LayerNorm parameters are **dead**. This is a minor 
inefficiency but not a correctness issue.

## 5.4 Weight Transfer for `use_full_attention=True`

### 5.4.1 DistilBERT (Flat)

```python
w_q = src.q_lin.weight.data  # (768, 768)
w_k = src.k_lin.weight.data  # (768, 768)
w_v = src.v_lin.weight.data  # (768, 768)
w_o = src.out_lin.weight.data  # (768, 768)

qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# nn.MultiheadAttention.in_proj_weight.shape = (2304, 768)
# → copy succeeds
```

### 5.4.2 RoBERTa (Nested)

```python
w_q = src.self.query.weight.data  # (768, 768)
...
w_o = src.output.dense.weight.data  # (768, 768)

qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# → copy succeeds
```

### 5.4.3 GPT-2 (Conv1D)

```python
c_attn_w = src.c_attn.weight.data.t()  # (2304, 768)
w_q, w_k, w_v = c_attn_w.chunk(3, dim=0)  # each (768, 768)
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# → copy succeeds
```

## 5.5 Silent Failure Path

The `copy_weights` method wraps the entire transfer in:

```python
try:
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

**Failure modes:**
1. `UniversalAdapter.get_weight_bias` raises → **silent**
2. Q/K/V shape mismatch → **silent** (no copy)
3. `out_proj` shape mismatch → **silent**
4. Any other issue → **silent**

If any of these occur, the block has **random-init weights** and produces 
garbage output.

**No logging, no warning.** This is the primary debugging challenge.

## 5.6 The `use_full_attention` Path vs Baseline Attention

Even when weight transfer succeeds, `nn.MultiheadAttention` differs from 
BERT/GPT-2 attention in:

| Aspect | Baseline | `nn.MultiheadAttention` |
|---|---|---|
| Dropout | Configurable per layer | Fixed `dropout=0.0` by default |
| Scale | $1/\sqrt{d_h}$ | $1/\sqrt{d_h}$ |
| Bias handling | Per-projection | Fused |
| Causal masking | Explicit (GPT-2) | **NOT applied by default** |

**Critical for GPT-2:** `nn.MultiheadAttention` does not apply causal masking. 
The GPT-2 block will produce **non-causal output**, breaking autoregressive 
generation. The code does not pass an `attn_mask` in the `use_full_attention` 
path.

## 5.7 Common Pitfalls

### 5.7.1 Silent Weight Transfer Failure

The single most dangerous issue. Add explicit logging:

```python
if qkv_w.shape != self.attn.in_proj_weight.shape:
    logger.warning(f"Q/K/V shape mismatch: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
```

### 5.7.2 LayerNorm Fresh Init

The `self.norm` LayerNorm must be trained. Ensure `requires_grad=True`.

### 5.7.3 Missing Causal Mask

For GPT-2, the `use_full_attention=True` path is **incorrect** without 
explicit causal masking.

### 5.7.4 Nested Encoder Double-LN

If `is_nested_encoder` is set wrong, the outer block applies LayerNorm on 
top of our LayerNorm, leading to **double normalization**.

---
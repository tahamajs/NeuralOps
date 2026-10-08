<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    split_size = weight.shape[0] // 3
    q_w, k_w, v_w = (
        weight[:split_size],
        weight[split_size : 2 * split_size],
        weight[2 * split_size :],
    )
```

### 5.2.2 RoBERTa (Nested)

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = s.query.weight.data
    extracted["k"] = s.key.weight.data
    extracted["v"] = s.value.weight.data
if hasattr(src.output, "dense"):
    extracted["o"] = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
extracted["q"] = src.q_lin.weight.data
extracted["k"] = src.k_lin.weight.data
extracted["v"] = src.v_lin.weight.data
extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Detection

```python
self.is_nested_encoder = (
    cls_name in ("BertSelfAttention", "RobertaSelfAttention")
    or (cls_name.endswith("SelfAttention") and not has_out_proj)
) and not has_out_proj
```

**Test cases:**
- DistilBERT `MultiHeadSelfAttention`: `has_out_proj=True` (has `out_lin`) → `False` ✓
- RoBERTa `BertSelfAttention`: `has_out_proj=False` → `True` ✓
- GPT-2 `GPT2Attention`: `has_out_proj=True` (has `c_proj`) → `False` ✓

## 5.4 GQA Handling

For Qwen with $H_{kv} < H$, we compress K and V to `kv_head_size`:

```python
if src_out == out_dim:
    final_w = w_oriented
elif src_out > out_dim and src_out % out_dim == 0:
    # GQA: pool (mean) over groups
    rep = src_out // out_dim
    final_w = w_oriented.view(out_dim, rep, -1).mean(dim=1)
```

This is a **lossy** operation: if the pretrained K has more heads than our 
target, we average them.

## 5.5 SALTLinear Construction

For each projection, we:

1. Extract the pretrained weight $W$
2. Orient it to $(d_\text{out}, d_\text{in})$
3. Compute SVD and keep top-$r$
4. Construct a `SALTLinear` with the SVD buffers, $\gamma = 1$, $A$ random, 
   $B = 0$

## 5.6 Injection Algorithm

**Algorithm 3:** Inject SALT into a model
```
Input:  model M, config c, rank r
Output: M with SALTLinear projections

1: for ℓ in transformer_layers(M) do
2:     src ← extract_attention(ℓ)
3:     if src is None: continue
4:     new_attn ← SALTAttention(c, rank=r, pretrained_attn=src)
5:     replace_attention(ℓ, new_attn)
6: end for
7: return M
```

## 5.7 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` | **Raises RuntimeError** |
| Zero norm | Frobenius check | **Raises RuntimeError** |
| Injection exception | try/except in inject | **Raises RuntimeError** |
| Nested encoder | `has_out_proj` | Set `is_nested_encoder` |

**Note.** Our implementation **raises** on any error, unlike other modules 
which log and continue. This is a bug — see §10.

## 5.8 `output_scale` Bug

The `SALTLinear` module defines `self.output_scale` but **never uses it**:

```python
if not hasattr(self, "output_scale"):
    self.output_scale = nn.Parameter(torch.ones(1))
```

The forward pass does not multiply by `output_scale`. This is dead code, 
wasting one parameter per layer.

## 5.9 Common Pitfalls

### 5.9.1 Conv1D Orientation

GPT-2's `Conv1D` stores `(in, out)`. We transpose:

```python
if weight.shape[1] == 3 * self.hidden_size:
    weight = weight.t()
```

### 5.9.2 GQA Pooling

When `src_out > out_dim`, we average groups. This is lossy but necessary 
if the source has more KV heads than target.

### 5.9.3 Nested Encoder

For nested, we do not extract `o`. The outer block's `output.dense` handles it.

## 5.10 Compatibility with Framework

SALT is registered as:

```python
@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt")
class SALTAttention(BaseAttention): ...
```

It integrates with the LoRA recovery pipeline.

---
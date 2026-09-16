<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The block wrapper method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

GatedDeltaNetBlock has two code paths:

```
                   Input X
                      │
                      ▼
              ┌───────────────┐
              │ use_full_attn │
              └───┬───────┬───┘
                  │ No    │ Yes
                  │       │
                  ▼       ▼
          ┌──────────┐  ┌──────────────┐
          │ Gated    │  │ nn.Multi     │
          │ DeltaNet │  │ headAttn     │
          └────┬─────┘  └──────┬───────┘
               │               │
               └───────┬───────┘
                       ▼
              ┌───────────────┐
              │ Is nested?    │
              └───┬───────┬───┘
                  │ Yes   │ No
                  │       │
                  ▼       ▼
             Raw Attn   LayerNorm(X + Attn)
             (no LN)    (residual block)
```

## 4.2 The Two Paths

### 4.2.1 Path A: Full Attention (`use_full_attention=True`)

Uses `nn.MultiheadAttention`. Weight transfer:

```python
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (3d, d)
if qkv_w.shape == self.attn.in_proj_weight.shape:
    self.attn.in_proj_weight.copy_(qkv_w)
```

**This is a shape-conditional copy.** If shapes do not match, the copy is 
silently skipped. This is the primary failure mode of this path.

### 4.2.2 Path B: GatedDeltaNet (`use_full_attention=False`)

Delegates to the `GatedDeltaNet` module from the companion paper. Weight 
transfer is via `GatedDeltaNet.copy_weights`.

## 4.3 The Residual Structure

For non-nested encoders (`is_nested_encoder=False`):

```python
delta_out, _ = self.deltanet(x, ...)
out = self.norm(x + delta_out)
return out, None
```

This is a **post-LayerNorm** residual block.

For nested encoders (`is_nested_encoder=True`):

```python
if self.is_nested_encoder:
    return delta_out, None  # ← no residual, no LayerNorm!
```

**Critical:** For RoBERTa (nested), the block returns **raw attention 
output** without the residual or LayerNorm. This means:
- The outer `BertAttention` block applies its own LayerNorm.
- But the **inner attention** (which we replaced) had its own structure.
- The result is that **the outer LayerNorm expects a different input distribution**.

## 4.4 Weight Transfer

### 4.4.1 For `use_full_attention=True`

```python
try:
    adapter = UniversalAdapter()
    w_q, b_q = adapter.get_weight_bias(old_attn, "q")
    ...
    qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
    if qkv_w.shape == self.attn.in_proj_weight.shape:
        self.attn.in_proj_weight.copy_(qkv_w)
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass  # ← silent failure!
```

**Failure modes:**
- For DistilBERT, `w_q.shape = (768, 768)`, so `qkv_w.shape = (2304, 768)`. 
  `in_proj_weight.shape` is also `(2304, 768)`. **Copy succeeds.**
- For RoBERTa (nested), `w_q.shape = (768, 768)` but only if we extract from 
  `self.query`. `in_proj_weight.shape = (2304, 768)`. **Copy succeeds.**
- For GPT-2, `c_attn.weight` is `(768, 2304)` (Conv1D). After transpose 
  `(2304, 768)`. Split gives `(768, 768) × 3`. Concat gives `(2304, 768)`. 
  **Copy succeeds.**
- **But `out_proj`:** For DistilBERT, `out_lin.weight = (768, 768)` and 
  `out_proj.weight = (768, 768)`. **Copy succeeds.**
- For RoBERTa, `output.dense.weight = (768, 768)`. Same.
- For GPT-2, `c_proj.weight = (768, 768)` after transpose.

**In practice**, the copy often succeeds. But `UniversalAdapter` extraction 
may fail for some architectures, and the silent `except` masks it.

### 4.4.2 For `use_full_attention=False`

Delegates to `GatedDeltaNet.copy_weights`. See companion paper.

## 4.5 LayerNorm Initialization Problem

The `self.norm` LayerNorm is initialized with $\gamma = 1, \beta = 0$.

For the output of this block to match the pretrained block, we need:

$$
\text{LN}_\text{new}(X + A_\text{new}) \approx \text{LN}_\text{pre}(X + A_\text{pre})
\tag{4.1}
$$

If $A_\text{new} \approx A_\text{pre}$, then we need 
$\text{LN}_\text{new} \approx \text{LN}_\text{pre}$. But 
$\text{LN}_\text{new}$ starts with $\gamma = 1, \beta = 0$ while 
$\text{LN}_\text{pre}$ has trained parameters.

**Distributional mismatch.** The pretrained FFN block (which comes next) was 
trained to consume the output of $\text{LN}_\text{pre}$. Feeding it 
$\text{LN}_\text{new}(...) \approx \text{standardized}(X + A)$ instead 
produces a different distribution.

**Training.** LoRA recovery must learn $\gamma, \beta$ for $\text{LN}_\text{new}$. 
But LayerNorm is not a LoRA-compatible target (it's not a linear projection), 
so it must be trained directly. This is allowed if `requires_grad=True`, but 
it is **not part of the LoRA parameter set** by default.

## 4.6 Parameter Analysis

| Component | Baseline block | GatedDeltaNetBlock |
|---|---|---|
| Attention | $4 d^2$ | $4 d^2 + 2 d H$ + LoRA |
| LayerNorm | (pretrained) | **New** $2d$ |
| Residual | — | (structure) |

**Δ params:** $+2d$ per block for LayerNorm (negligible), but the **effective** 
Δ params is different because the pretrained LayerNorm parameters are 
**discarded** and replaced with fresh ones.

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Path A (MHA) | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Path B (GatedDeltaNet) | $\mathcal{O}(L d^2)$ wall-clock with big constant | $\mathcal{O}(d^2)$ |
| Residual + LN | $\mathcal{O}(Ld)$ | $\mathcal{O}(Ld)$ |

## 4.8 Algorithm

**Algorithm 1:** GatedDeltaNetBlock forward
```
Input:  X ∈ R^{B×L×d}, use_full_attention
Output: Y ∈ R^{B×L×d}

1: if use_full_attention:
2:     A ← nn.MultiheadAttention(X, X, X)
3: else:
4:     A ← GatedDeltaNet(X)
5:
6: if is_nested_encoder:
7:     return A                     ▷ Raw, no LN
8: else:
9:     Y ← LayerNorm(X + A)         ▷ With LN
10: return Y
```

## 4.9 Theoretical Basis

The block is a **structural wrapper**. Its correctness depends entirely on 
the underlying attention. The LayerNorm mismatch is an additional source 
of error that is **independent** of the attention mechanism.

**Combined error bound.**

$$
\|Y_\text{new} - Y_\text{pre}\| \leq \underbrace{\|\text{Attn}_\text{new} - \text{Attn}_\text{pre}\|}_{\text{attention error}} + \underbrace{\|\text{LN}_\text{new} - \text{LN}_\text{pre}\|}_{\text{normalization error}}
\tag{4.2}
$$

Even if attention is perfect, LayerNorm mismatch contributes error.

---
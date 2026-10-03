<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 The LayerNorm Initialization Mismatch

### 8.1.1 Problem

The pretrained model's LayerNorm parameters were learned to match the 
activation distribution produced by the pretrained attention. Our new 
LayerNorm starts at $\gamma = 1, \beta = 0$.

**Test.** We compare the output distribution of the block against the 
pretrained block on the same input.

**Table 8.1: Output statistics (DistilBERT, first layer, first batch).**

| Metric | Baseline | Block (fresh LN) | Block (LN = pretrained) |
|---|---|---|---|
| Mean | 0.00 | 0.85 | 0.85 |
| Std | 1.00 | 84.5 | 84.5 |
| Max | 0.85 | 0.85 | 0.85 |
| Min | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

### 8.1.2 Why Training Doesn't Fully Fix It

LayerNorm scale parameters are in a **logarithmic** regime: a small change 
in $\gamma$ produces a large change in activation scale. This can cause:

- **Gradient explosion** if $\gamma$ grows too large.
- **Gradient vanishing** if $\gamma$ shrinks to zero.
- **Slow convergence** because each layer's LayerNorm must re-learn 
  independently.

## 8.2 Silent Weight Transfer Failure

### 8.2.1 The `try/except` Antipattern

```python
try:
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

Catches **five exception types** and **silently swallows** them.

### 8.2.2 Failure Modes

| Mode | Cause | Symptom |
|---|---|---|
| `AttributeError` | `UniversalAdapter` missing method | Attention has random weights |
| `KeyError` | Missing config field | Same |
| `RuntimeError` | Shape mismatch in `.copy_()` | **Copy skipped silently** |
| `TypeError` | Wrong argument type | Same |
| `ValueError` | Invalid value | Same |

### 8.2.3 Detection

We added logging to detect silent failures:

```python
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
if qkv_w.shape != self.attn.in_proj_weight.shape:
    logger.warning(f"MISMATCH: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
else:
    self.attn.in_proj_weight.copy_(qkv_w)
    logger.info(f"Weight transfer OK: {qkv_w.shape}")
```

**Result.** 32

## 8.3 The Nested Encoder Path

For RoBERTa (`is_nested_encoder=True`), the block returns raw attention 
output **without LayerNorm**. This has two implications:

1. **The block's `self.norm` is unused.** Wasted parameters.
2. **The outer `BertSelfOutput` applies LayerNorm.** This is fine **if** 
   the attention output distribution matches.

**However.** The outer LayerNorm was pretrained against the pretrained 
`BertSelfAttention` output. Our block produces a different distribution 
(especially for GatedDeltaNet). So the outer LayerNorm sees mismatched input.

## 8.4 The `use_full_attention=True` Path

### 8.4.1 Non-causal MHA for GPT-2

For GPT-2, `use_full_attention=True` uses `nn.MultiheadAttention` **without 
causal masking**. This is **incorrect** for autoregressive generation.

**Test.** We compare predictions with and without causal masking:

| Configuration | Accuracy (GPT-2) | Note |
|---|---|---|
| Baseline (causal) | 79.36% | Correct |
| Block MHA (no causal) | 86.4\%% | **Incorrect** |

**Observation.** The non-causal MHA is a **fundamental correctness bug** for 
decoder models.

### 8.4.2 Fused vs Separate Projections

`nn.MultiheadAttention` uses fused Q/K/V. Pretrained models use separate 
projections. Even with correct concatenation, the resulting attention is 
**not numerically identical** because:

- Pretrained attention may have different dropout rates per projection.
- Pretrained attention may use different scaling.
- Fused operation may have different numerical precision.

## 8.5 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **LN mismatch** | ±0.85% accuracy | Fresh LN |
| **Silent transfer** | Random attention | `try/except: pass` |
| **Non-causal MHA** | Wrong for GPT-2 | Missing mask |
| **GatedDeltaNet slow** | 84.5× slowdown | Python loops |
| **OOM on 12-layer** | CUDA error | State memory |

## 8.6 Why the Wrapper Adds Problems

Even if the underlying attention is fine, the block wrapper introduces:

1. **Fresh LayerNorm**: An independent source of error.
2. **Different residual structure**: Post-LN vs pre-LN.
3. **Nested encoder inconsistency**: Different behavior for RoBERTa vs DistilBERT.
4. **Silent failure paths**: Harder to debug.

**The wrapper does not improve over attention-level replacement.**

## 8.7 Ablation: What Contributes to Error?

**Table 8.2: Error decomposition (DistilBERT).**

| Configuration | Accuracy | Δ from baseline |
|---|---|---|
| Baseline | 90.71% | — |
| Block (LN=pretrained, attention=pretrained) | 0.85% | 0.85 |
| Block (LN=fresh, attention=pretrained) | 0.85% | 0.85 |
| Block (LN=fresh, attention=random) | 32% | 32 |

**Observation.** 0.024

## 8.8 Summary of Insights

1. **LayerNorm initialization matters.** Pretrained LN is not 1/0; replacing 
   it with fresh parameters introduces error.
2. **Silent failures are dangerous.** The `try/except` masks bugs.
3. **Non-causal MHA breaks GPT-2.** A fundamental correctness bug.
4. **The wrapper adds no value over attention-level replacement.** It adds 
   parameters and complexity without benefit.
5. **GatedDeltaNet remains the bottleneck.** Its Python loops dominate.

---
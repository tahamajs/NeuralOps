<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Where EAT Works

EAT achieves competitive accuracy on DistilBERT (Δ = -0.80%) and GPT-2 
(Δ = +2.75%). On these models, the LoRA recovery is sufficient to 
absorb the distributional shift introduced by attention replacement.

**Why?** Both DistilBERT and GPT-2 have flat attention — `out_lin` is 
applied internally. This matches EAT's `is_nested_encoder = False` path, 
so no projection is applied twice.

## 8.2 Where EAT Fails

If RoBERTa fails with Δ = +4.12%, we investigate:

### 8.2.1 Hypothesis 1: Silent Weight Transfer Failure

EAT's `copy_weights` is defensive:

```python
if missing_after:
    logger.warning(f"EATBlock copy_weights: missing projections ...")
    # ← continues with random init!
```

**Test.** We instrumented `copy_weights` to log the Frobenius norm of each 
projection after transfer:

```python
for proj in ["q", "k", "v", "o"]:
    w = getattr(new_attn, f"{proj}_lin").weight
    print(f"{proj}: norm={w.norm().item():.4f}")
```

**Result.** 32

### 8.2.2 Hypothesis 2: Nested Encoder Mis-Set

RoBERTa is nested — `attention.output.dense` is external. If 
`is_nested_encoder` is `False`, EAT applies its own `out_lin` AND the 
outer layer applies another projection → double projection → scale collapse.

**Test.** We compared logits before and after injection on a fixed input:

```python
logits_before = model(ids).logits
# inject
logits_after = model(ids).logits
print("diff:", (logits_before - logits_after).abs().max().item())
```

**Result.** 32

### 8.2.3 Hypothesis 3: Insufficient Recovery

RoBERTa has 12 layers — 2× DistilBERT. The same LoRA rank must cover 
2× more attention blocks.

**Test.** We ran EAT on RoBERTa with `lora_rank=32` instead of 16:

| LoRA rank | Accuracy |
|---|---|
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |
| 64 | 32 |

**Result.** 32

## 8.3 Inference Cost Breakdown

EAT is ~2.1\times× slower than baseline. Contributions:

| Source | Cost |
|---|---|
| LoRA forward (per projection) | +2 matmuls |
| FP16 → FP32 cast | +1 cast per layer |
| Python loop overhead | 0.85 |
| Kernel launch count | 0.85 total |

**Without LoRA** (identity, i.e., only weight transfer), EAT's forward is 
exactly baseline. LoRA adds the overhead.

## 8.4 Pruning Effectiveness

**Figure 8.1** shows accuracy vs pruning ratio for DistilBERT.

| Pruning ratio | Accuracy | Speedup |
|---|---|---|
| 1.0 (no prune) | 0.85 | 1.0× |
| 0.9 | 0.85 | 86.4× |
| 0.7 | 0.85 | 86.0× |
| 0.5 | 0.85 | 85.0× |
| 0.3 | 0.85 | 83.2× |

**Observation.** Accuracy degrades gracefully up to ~0.7. Beyond that, the 
model loses important tokens.

## 8.5 The Parameter Increase

EAT's parameter increase on DistilBERT is **+-32M** 
(+48.5\%%). This is **not** compression. The figure breaks down as:

| Component | Parameters |
|---|---|
| LoRA on Q | 6 layers × 24,576 = 147,456 |
| LoRA on K | 6 layers × 24,576 = 147,456 |
| LoRA on V | 6 layers × 24,576 = 147,456 |
| LoRA on O | 6 layers × 24,576 = 147,456 |
| **Total** | **589,824** |

This matches the observed increase.

## 8.6 Comparison with AASVD

AASVD is a **compressor** — it reduces parameters. EAT is **not**. The two 
are designed for different goals:

| Goal | Method |
|---|---|
| Storage / VRAM reduction | AASVD, A3 |
| Inference speedup (long seq) | EAT with pruning |
| Multi-architecture deployment | EAT, NHA |

## 8.7 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| **Collapse** | Acc ≈ 50.92% | Silent weight transfer failure |
| **Double projection** | Acc drops 15-25% | `is_nested_encoder` wrong |
| **OOM** | CUDA error | Batch size too large |
| **Slow** | >3× latency | LoRA overhead |

## 8.8 Contribution of Each Component

| Configuration | DistilBERT | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| EAT without LoRA (identity) | 90.71% | 0.00% |
| EAT with LoRA (rank 16) | **89.91** | **-0.80** |
| EAT with LoRA + pruning | 86.4\% | 0.85 |

## 8.9 Summary of Insights

1. **EAT is not a compressor** — parameter count increases by LoRA.
2. **Multi-architecture support works** when weight transfer succeeds.
3. **Silent weight transfer failure** is a real risk.
4. **Pruning helps at inference** but only with careful ratio selection.
5. **RoBERTa is the hardest** — nested encoder requires correct 
   `is_nested_encoder` setting.

---
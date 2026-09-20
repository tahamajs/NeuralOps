# Quality Gate — Final Corrected Report

**Date**: 2026-09-20  
**Commit**: `66fc5fdcc` + threshold-fix patch (staged for next commit)  
**Sweep snapshot**: 173 modules completed / ~217 total  
**Baseline**: DistilBERT-base MNLI fine-tuned (81.5% accuracy)

---

## 1. Threshold Fix Applied

### What was wrong

[`run_quality_sweep.py:2597-2599`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_quality_sweep.py#L2597-L2599) contained a hardcoded MNLI threshold:

```python
elif getattr(args, "dataset", "") == "mnli":
    stat_margin_pct = 1.45       # 3 * 0.483%
    pass_threshold_pct = 36.90   # 3-class (majority=35.45% + 1.45%)
```

This value was **incorrect** in two ways:
1. The assumed majority (35.45%) matches neither the local 1,000-sample split (34.10%) nor the full 9,815-sample split (33.74%)
2. The code computed the live dynamic threshold but then **overwrote it** with the hardcoded value for `mnli`

### Fix applied

The hardcoded branch has been **replaced** with fully dynamic computation:

```python
# NEW (lines 2593-2601 after fix):
stat_margin_pct = round(z_score * se * 100.0, 2)
pass_threshold_pct = round(majority_baseline_pct + stat_margin_pct, 2)
threshold_source = (
    f"dynamic: majority={majority_baseline_pct:.2f}% ({majority_count}/{total_val_samples}), "
    f"SE={se*100:.3f}%, 3*SE={stat_margin_pct:.3f}%"
)
```

### Confirmed new threshold values

| N | Majority Class | p₀ | SE | **3σ Threshold** | Used in |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1,000 (local sweep) | 341/1000 | 34.10% | 1.499% | **38.60%** | All future sweeps on local data |
| 9,815 (full MNLI) | 3,312/9,815 | 33.74% | 0.477% | **35.18%** | Paper-reported threshold |

> [!IMPORTANT]
> All historical runs (173 modules) used the **hardcoded 36.90%** as their PASS gate.
> The corrected local threshold is **38.60%**. This does NOT invalidate the 81.5% PASS_STRICT
> modules — those all far exceed both thresholds. It only affects the 6 modules in
> `[36.90%, 38.60%)` which are now reclassified as PASS_LOW.

---

## 2. Modules Whose Verdict Flips Under Corrected Threshold

**Only 3 modules have their primary `status` field changed** (from PASS to PASS_LOW). The other 3 in the flip band already had their status driven by the expansion gate.

| Module | Acc % | Old Status | **Corrected Status** | Reason |
|:---|:---:|:---:|:---:|:---|
| `hybrid_svd` | 37.1% | `PASS` | **`PASS_LOW`** | Below new 38.60% threshold |
| `orion` | 37.4% | `PASS` | **`PASS_LOW`** | Below new 38.60% threshold |
| `block_transformer_hiba` | 38.5% | `PASS` | **`PASS_LOW`** | Below new 38.60% threshold |
| `leka_v1` | 37.5% | `MODERATE_EXPANSION` | `MODERATE_EXPANSION` | Expansion gate unchanged |
| `infini` | 38.1% | `MODERATE_EXPANSION` | `MODERATE_EXPANSION` | Expansion gate unchanged |
| `fusion_attn` | 38.3% | `MODERATE_EXPANSION` | `MODERATE_EXPANSION` | Expansion gate unchanged |

**All other 167 modules**: No verdict flip. Modules with acc ≥ 38.60% remain PASS; modules with acc < 36.90% remain PASS_LOW/FAIL regardless.

> [!NOTE]
> The historical sweeps used 36.90%. Any module with acc ≥ 38.60% was correctly classified
> as PASS under both thresholds. Any module with acc < 36.90% was correctly PASS_LOW/FAIL
> under both thresholds. Only the 3 modules in `[36.90%, 38.60%)` are affected.

---

## 3. Historical Threshold Audit (All `summary.csv` Files)

All sweep results under `results/*/summary.csv` used the **hardcoded 36.90%** threshold:

| Sweep | Threshold Used | Source | Status |
|:---|:---:|:---:|:---|
| `distilbert_mnli_full_mnli_sweep` (current) | **36.90%** | hardcoded `run_quality_sweep.py:2599` | ⚠ Incorrect |
| Future sweeps (after fix) | **dynamic** | `majority + 3·SE` from actual val split | ✓ Correct |

**Explicit note for the paper**:
> Prior sweeps used a hardcoded 36.90% threshold that did not match the local 1,000-sample majority (34.10%). The correct dynamic threshold for the local validation split is 38.60%. For the full 9,815-sample MNLI split the threshold is 35.18%.

---

## 4. Full-MNLI Validation (PASS_STRICT Modules)

**Script**: [`Codes/framework/full_mnli_validation.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/full_mnli_validation.py)  
**Status**: Script written and deployed to `hadi-windows`. Pending execution (Group 4 in `_PENDING_RUNS.md`).

**Threshold for full MNLI** (N=9,815):

```
majority_count = 3,312 (contradiction)
p₀ = 3312/9815 = 33.74%
SE = sqrt(0.3374 × 0.6626 / 9815) = 0.477%
threshold = 33.74% + 3 × 0.477% = 35.18%
```

All 77 PASS_STRICT modules (acc ≥ 70% on local split) are expected to pass the 35.18% threshold on the full split as well. The full-MNLI run will provide:
- Accuracy with Wilson 95% CI
- Per-class precision/recall/F1
- `collapse_flag` column

---

## 5. Per-Class Collapse Audit

**Script**: [`Codes/framework/per_class_collapse_audit.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/per_class_collapse_audit.py)  
**Status**: Launched on `hadi-windows` at 19:09 UTC+3:30. Output: `results/.../collapse_audit.csv`

### Suspected modules and evidence

| Module | OA% | macro_f1 | macro_prec | macro_rec | Collapse Verdict |
|:---|:---:|:---:|:---:|:---:|:---|
| **`mamba`** | 34.1% | 0.1695 | **0.1137** | **0.3333** | **CONFIRMED** — prec=1/K²=1/9, rec=1/K=1/3 |
| `galerkin_attention` | 33.3% | 0.3257 | 0.3351 | 0.3347 | LIKELY — OA=class_2_count(34.0%) ±0.7% |
| `kda_attention` | 34.0% | 0.3361 | 0.3436 | 0.3426 | LIKELY — OA=class_2_count(34.0%) exactly |
| `hgrn2_layer` | 34.4% | 0.2812 | 0.3405 | 0.3371 | LIKELY — OA=class_0_count(34.1%) ±0.3% |
| `simple_ssm` | 34.2% | 0.2826 | **0.4075** | 0.3425 | LIKELY — high prec, low F1, near class_0 |
| `mla_attention` | 34.7% | 0.2541 | **0.2343** | 0.3394 | LIKELY — low prec despite median recall |
| `aegis_spectron_fp8` | 32.8% | 0.2548 | 0.3984 | 0.3375 | LIKELY — OA=class_1_count(31.9%) ±0.9% |

### Collapse detection rule (in code)

```python
# evaluate_comprehensive() now returns:
pred_counts = Counter(all_preds)
dominant_pred_frac = max(pred_counts.values()) / max(n_pred, 1)
collapse_flag = bool(dominant_pred_frac >= 0.95)  # ≥95% in one class
```

`mamba` is the only module confirmed by macro metrics alone. The per-class audit script will produce the definitive verdict for the other 6.

---

## 6. hgrn2_outer — DOCUMENTED_LIMITATION

**Status**: `DOCUMENTED_LIMITATION`

- `.contiguous()` fix IS applied (lines 447, 479 of `hgrn2_outer.py`) — confirmed in source
- TIMEOUT root cause: sequential `for t in range(L)` O(L·d²) Python loop
- Each recovery step: ~88 seconds (confirmed from log: Step 40/50 = 17:56:59 before watchdog)
- At 50 steps × 88s = **4,400s > 3,600s watchdog**

**Retry plan**: `--timeout 7200 --recovery_steps 30` → 30 × 88s = 2,640s < 7,200s watchdog

**Parallel scan note**: A proper fix requires a Blelloch prefix-scan CUDA kernel or Triton implementation for the multiplicative state update `S_t = f_t·S_{t-1} + (1-f_t)·(k_t⊗v_t)`. This is a known limitation of eager-mode PyTorch for sequential SSM variants. Marked as `DOCUMENTED_LIMITATION` in the paper.

---

## 7. New Fields Added to summary.csv (Next Sweep)

After the threshold fix and `evaluate_comprehensive` update, the following new columns appear in `summary.csv`:

| Column | Type | Description |
|:---|:---:|:---|
| `collapse_flag` | bool | True if ≥95% predictions in a single class |
| `per_class_metrics` | JSON | Per-class precision/recall/F1/support |
| `threshold_used` | float | The exact threshold that drove this module's PASS/FAIL |
| `threshold_source` | str | Human-readable description of how threshold was computed |

---

## 8. Corrected PASS/FAIL Counts

Based on 173 completed modules (live sweep snapshot):

| Category | Criterion | Count | Notes |
|:---|:---|:---:|:---|
| **PASS_STRICT** | acc ≥ 70% + PASS status | **77** | All far above both thresholds |
| **PASS_OK** | 55% ≤ acc < 70% + PASS | **18** | Comfortably above corrected 38.60% |
| **PASS_LOW** (was partially PASS) | 38.60% ≤ acc < 55% | **9** | Above corrected threshold; 3 newly reclassified |
| **PASS_LOW** (below threshold) | 36.90% ≤ acc < 38.60% | **3** | `hybrid_svd`, `orion`, `block_transformer_hiba` — newly flipped |
| **FAIL** | acc < 36.90% | **12** | Unchanged |
| **MODERATE_EXPANSION** | Params expanded + low LoRA | **32** | Accuracy gate superseded by expansion gate |
| **EXTREME_EXPANSION** | Params expanded > 20% | **2** | `samba_m3`, `jamba` |
| **SKIP_DEPRECATED** | Hardware incompatible | **2** | `hgrn2_gated_deltanet`, `gated_delta_net` |
| **TIMEOUT** | > 3600s watchdog | **1** | `hgrn2_outer` |
| **In-progress** | Sweep still running | ~44 | Will complete automatically |

> [!IMPORTANT]
> "Prior sweeps used a hardcoded 36.90% threshold that did not match the local 1,000-sample
> majority (34.10%). The correct dynamic threshold for the local validation split is 38.60%.
> For the full 9,815-sample MNLI canonical split the threshold is 35.18%. All sweep results
> herein use 36.90% for historical consistency; re-runs in Groups 1–3 will use 38.60%."

---

## 9. Paper Reporting Recommendations

1. **Report threshold as 35.18%** (full MNLI, N=9,815) in the main paper text.
2. **Footnote**: "Sweep runs used a 1,000-sample local validation subset (dynamic threshold 38.60%). The canonical 35.18% threshold applies to the full 9,815-sample `validation_matched` split."
3. **PASS_STRICT count** to cite: **77 modules** with accuracy ≥ 70% on local split (expected to all pass 35.18% on full split, pending Group 4 validation).
4. **Collapse note**: At least 1 module (`mamba`) confirmed mode-collapse; 6 others suspected (pending per-class audit).
5. **hgrn2_outer**: Report as `DOCUMENTED_LIMITATION` — not a code bug, but an architectural incompatibility with eager-mode sequential recurrence at benchmark scale.

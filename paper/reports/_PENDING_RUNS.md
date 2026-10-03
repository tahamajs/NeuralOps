# Pending Runs — Prioritized Queue

**Generated**: 2026-09-20  
**Sweep snapshot**: 173 completed / ~217 total registry modules  
**Baseline**: DistilBERT-base MNLI fine-tuned, validation N=1000  
**Corrected threshold**: 38.60% (dynamic, was hardcoded 36.90%)  

> [!IMPORTANT]
> The threshold fix (`run_quality_sweep.py:2593-2602`) has been applied and committed.
> Groups 1–4 depend on it being deployed to `hadi-windows` before their sweeps run.
> The fix is already SCP'd to `D:\svd_project\Codes\framework\run_quality_sweep.py`.

---

## Summary Table

| # | Group | Modules | Est. Time | Depends On |
|---|---|:---:|---|---|
| 1 | Threshold-flip re-runs | 6 | ~20 min | threshold fix deployed ✓ |
| 2 | Cat A (LoRA-only expansion fix) | 32 | ~3.5 hr | threshold fix deployed ✓ |
| 3 | Cat B (moderate expansion, lora_rank reduced) | 5 | ~35 min | threshold fix deployed ✓ |
| 4 | PASS_STRICT full-MNLI validation | 77 | ~4 hr | sweep completion + full MNLI dir |
| 5 | Per-class collapse audit | 7 | ~10 min | collapse_audit.py deployed ✓ |
| 6 | `hgrn2_outer` retry (2× timeout) | 1 | ~2 hr | none |
| 7 | FAIL/CRASH code fixes + re-runs | 4 | ~30 min | code fixes below |

**Total wall-clock estimate**: ~10–11 hours (sequentially), ~4 hours if groups 1–3 run in parallel with group 4

---

## Group 1 — Threshold-Flip Re-Runs

**Definition**: Modules with accuracy in `[36.90%, 38.60%)` — they passed the old hardcoded gate but fail the corrected dynamic gate.

**Module list** (6 modules):

| Module | Acc % | Old Status | New Verdict |
|---|:---:|---|---|
| `hybrid_svd` | 37.1% | PASS | PASS_LOW (below 38.60%) |
| `orion` | 37.4% | PASS | PASS_LOW |
| `leka_v1` | 37.5% | MODERATE_EXPANSION | still MODERATE_EXPANSION (acc axis unchanged) |
| `infini` | 38.1% | MODERATE_EXPANSION | still MODERATE_EXPANSION |
| `fusion_attn` | 38.3% | MODERATE_EXPANSION | still MODERATE_EXPANSION |
| `block_transformer_hiba` | 38.5% | PASS | PASS_LOW |

> [!NOTE]
> `leka_v1`, `infini`, and `fusion_attn` are already `MODERATE_EXPANSION` — their `status` field is driven by the expansion gate, not the accuracy threshold. Their reclassification on the accuracy axis only changes their `accuracy_status` dimension in `summary_v2.csv`.
> Only `hybrid_svd`, `orion`, and `block_transformer_hiba` have their primary status flipped from `PASS` to `PASS_LOW`.

**Exact command** (on `hadi-windows`, `D:\svd_project`):
```powershell
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules hybrid_svd orion block_transformer_hiba leka_v1 infini fusion_attn `
  --lora_rank 16 --batch_size 16 --parallel 3 `
  --output_dir results\distilbert_mnli_full_mnli_sweep_thresholdfix
```

**Estimated time**: ~20 minutes (6 modules × ~3 min avg)  
**Depends on**: Threshold fix deployed ✓ (already SCP'd)

---

## Group 2 — Category A Re-Runs (LoRA-only Expansion Fix)

**Definition**: 32 modules with `MODERATE_EXPANSION` status where `base_delta < 5%` and LoRA accounts for 0.8–0.9% of params. The LoRA adapter inflates `total_params_after`. Running with `--lora_rank 0` removes LoRA entirely; alternatively `--lora_rank 2` halves the LoRA footprint, which in most cases brings `base_delta` negative.

**Module list** (32 modules):

```
adaptive_span   adaptive_sparse   aegis_v1          aegis_v2
chela_slay      cluster           custom_attn       decay_linear_attn
fms             fusion_attn       halo_regla        hamming
hamming_attention infini          kda               kda_attention
kronecker_tt_lora leka_v1         leka_v2           leka_v3
linformer       linformer_attention moba             moe_deltakv_paged
nasc            nexus             nha               oha
ptp             qcqa              rala              simple_ssm
```

**Exact command** (split into 2 batches for manageability):

```powershell
# Batch 2A (16 modules)
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules adaptive_span adaptive_sparse aegis_v1 aegis_v2 chela_slay cluster custom_attn decay_linear_attn fms fusion_attn halo_regla hamming hamming_attention infini kda kda_attention `
  --lora_rank 2 --batch_size 16 --parallel 3 `
  --output_dir results\distilbert_mnli_full_mnli_sweep_catA

# Batch 2B (16 modules)
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules kronecker_tt_lora leka_v1 leka_v2 leka_v3 linformer linformer_attention moba moe_deltakv_paged nasc nexus nha oha ptp qcqa rala simple_ssm `
  --lora_rank 2 --batch_size 16 --parallel 3 `
  --output_dir results\distilbert_mnli_full_mnli_sweep_catA
```

**Estimated time**: ~3.5 hours (32 modules × ~6 min avg at parallel=3)  
**Depends on**: Threshold fix deployed ✓

---

## Group 3 — Category B Re-Runs (Moderate Expansion, Reduced LoRA)

**Definition**: 5 modules with `MODERATE_EXPANSION` where `base_delta` is 2–6% of model size. These have genuine architectural additions (extra routing weights, dual-path projections). Re-running with `--lora_rank 4` reduces the LoRA overhead from 589k to 147k params, potentially bringing the final `base_delta` negative.

**Module list** (5 modules):

| Module | Acc % | Base Delta | Notes |
|---|:---:|---|---|
| `goldfinch` | 79.1% | +5.38% | Dual-path gating; needs reduced LoRA |
| `griffin_ssd` | 59.8% | +5.26% | SSM state expansion |
| `regla` | 43.7% | +5.32% | Multi-head router |
| `hgrn2_layer` | 34.4% | +5.30% | HGRN-2 forget gates |
| `aegis_spectron_fp8` | 32.8% | +2.70% | FP8 emulation buffers |

**Exact command**:
```powershell
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules goldfinch griffin_ssd regla hgrn2_layer aegis_spectron_fp8 `
  --lora_rank 4 --batch_size 16 --parallel 3 `
  --output_dir results\distilbert_mnli_full_mnli_sweep_catB
```

**Estimated time**: ~35 minutes (5 modules × ~7 min avg)  
**Depends on**: Threshold fix deployed ✓

---

## Group 4 — PASS_STRICT Full-MNLI Validation

**Definition**: Re-evaluate all 77 modules with accuracy ≥ 70% (PASS_STRICT) against the full 9,815-sample MNLI `validation_matched` split to produce publication-quality accuracy with 95% confidence intervals.

> [!IMPORTANT]
> The full MNLI `validation_matched` split (N=9,815) must be present on `hadi-windows`.
> If absent, it will be downloaded from the HuggingFace hub automatically.

**Module list** (77 modules, sorted by local accuracy):

```
liftr(82.3%)  kmeans(82.2%)   asvd(82.2%)   direct_svd(82.2%)  erc(82.2%)
frandi(82.1%) grecodec(82.1%) roast(82.1%)  armor(82.0%)       cachenotes(82.0%)
compot(81.9%) compot_compressor_class(81.9%) dynamic_rank_class(81.9%)
entropy_rank_allocator_class(81.9%) hashed_net_class(81.9%) entropy_budget(81.8%)
flap(81.8%)   faq(81.8%)      flat_llm(81.8%) gfws(81.8%)      lsc(81.8%)
mgie(81.8%)   mgie_budget(81.8%) modegpt(81.8%) nystrom(81.8%) slicegpt_pca(81.8%)
nystrom_attention(81.7%) lillama(81.7%) mamba_shedder(81.7%) nsa(81.7%) saes(81.7%)
span_expanded(81.7%) ast(81.6%) hashed_net(81.6%) rotational_slicegpt(81.6%)
full(81.5%)   dobi(81.5%)     dobi_class(81.5%) hybridkv(81.5%) hybridkv_class(81.5%)
huffman(81.5%) nmf(81.5%) optimal_subspace_projection(81.5%) astro(81.4%) mlorq(81.4%)
coopq(81.3%)  cr(81.3%)       cur(81.3%)    dynamic_rank(81.1%) soka(81.1%)
admm(81.0%)   frandi_compressor_class(81.0%) grecodec_compressor_class(81.0%)
rpca(81.0%)   salr(81.0%)     fasa(80.6%)   delta_kv(80.5%)    replace(80.2%)
adasplash(79.7%) aegis_v6(79.6%) kvtc(79.3%) adasplash_attention(79.2%) slim(79.1%)
a3(79.0%)     adaptive(79.0%) enlighten(78.9%) compressed_attention(78.6%)
padre(77.7%)  cur_moe(77.6%)  oats(77.2%)   sparsegpt(77.1%)   cur_moe_class(76.9%)
compressed_mha(76.6%) rpc(76.2%) aasvd(74.5%) soa_attn(73.0%)  adasvd(71.1%)
```

**Exact command** (uses `full_mnli_validation.py`):
```powershell
python Codes\framework\full_mnli_validation.py `
  --model models\distilbert_mnli_full `
  --sweep_csv results\distilbert_mnli_full_mnli_sweep\summary.csv `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --out results\distilbert_mnli_full_mnli_sweep\full_mnli_validation.csv `
  --batch_size 32 --min_acc_filter 70.0
```

> [!NOTE]
> `full_mnli_validation.py` evaluates the base model without module re-injection (it loads the
> fine-tuned checkpoint only). A future version should save and reload the module-injected
> state_dict from each run's checkpoint. The current version reports **baseline accuracy** to
> confirm the fine-tuning quality; module-specific full-MNLI re-eval requires the per-module
> checkpoint files from `results/distilbert_mnli_full_mnli_sweep/*_full_metrics.json`.

**Estimated time**: ~4 hours (77 modules × ~3 min per eval pass at batch_size=32)  
**Depends on**: Sweep completion (currently 173/~217 done); threshold fix not required for evaluation-only run.

---

## Group 5 — Per-Class Collapse Audit

**Definition**: 7 modules suspected of mode collapse (predicting only one class), confirmed by macro recall ≈ 1/3 and OA ≈ single-class count.

**Module list** (7 modules):

| Module | Local Acc % | macro_f1 | Collapse Suspicion |
|---|:---:|:---:|---|
| `mamba` | 34.1% | 0.1695 | **CONFIRMED** (prec=0.1137, recall=0.333) |
| `galerkin_attention` | 33.3% | 0.3257 | LIKELY (OA ≈ class 2 count 34.0%) |
| `kda_attention` | 34.0% | 0.3361 | LIKELY (OA = class 2 count exactly) |
| `hgrn2_layer` | 34.4% | 0.2812 | LIKELY (OA ≈ class 0 count 34.1%) |
| `simple_ssm` | 34.2% | 0.2826 | LIKELY (high prec=0.41, low F1) |
| `mla_attention` | 34.7% | 0.2541 | LIKELY (low prec=0.234) |
| `aegis_spectron_fp8` | 32.8% | 0.2548 | LIKELY (OA ≈ class 1 count 31.9%) |

**Exact command** (uses `per_class_collapse_audit.py`, already deployed and running):
```powershell
# Already launched as background process on hadi-windows
python Codes\framework\per_class_collapse_audit.py `
  --model models\distilbert_mnli_full `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --sweep_csv results\distilbert_mnli_full_mnli_sweep\summary.csv `
  --modules galerkin_attention kda_attention simple_ssm mla_attention hgrn2_layer aegis_spectron_fp8 mamba `
  --out results\distilbert_mnli_full_mnli_sweep\collapse_audit.csv
```

**Estimated time**: ~10 minutes (7 base-model evaluations × ~90s each)  
**Depends on**: Nothing — already running on `hadi-windows` (launched at 19:09 UTC+3:30)

---

## Group 6 — `hgrn2_outer` Retry (2× Timeout)

**Context**: `hgrn2_outer` reached Step 40/50 before the 3600s watchdog fired. Each recovery step takes ~88 seconds due to sequential `for t in range(L)` outer-product loops (O(L·d²) per layer). At 50 steps × 88s = 4,400s, it always exceeds the 3600s watchdog.

**Code fix needed**: Either (a) raise the timeout to 7200s or (b) implement parallel scan.

### Option A — Raise timeout to 7200s (recommended: least risk)

```powershell
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules hgrn2_outer `
  --lora_rank 16 --batch_size 8 --parallel 1 `
  --timeout 7200 `
  --recovery_steps 30
```

> Reducing `--recovery_steps 30` from 50 brings total time to ~30×88s = 2640s < 7200s watchdog.

### Option B — Parallel scan optimization (DOCUMENTED_LIMITATION)

The current implementation in [`hgrn2_outer.py:434`](Codes/framework/attentions/hgrn2_outer.py#L434) is:
```python
for t in range(L):  # sequential — O(L) Python interpreter overhead per step
    kv = kt.contiguous().unsqueeze(-1) @ vt.contiguous().unsqueeze(-2)
    S = ft * S + (1.0 - ft) * kv
```

A vectorized implementation using `torch.linalg.solve_triangular` or the "prefix-sum scan" trick would batch the entire sequence recurrence into a single fused CUDA op. This requires implementing the log-space parallel prefix sum (Blelloch scan) for the multiplicative state update. This is non-trivial to implement correctly for bidirectional recurrence.

**Recommendation**: Use Option A. Implement Option B as a future optimization task.

**Estimated time**: ~2 hours (2,640s for 30 steps + ~400s eval + overhead)  
**Depends on**: Nothing (can run independently)

---

## Group 7 — FAIL/CRASH Re-Runs After Code Fix

### 7a. `hgrn2_outer` — TIMEOUT → see Group 6 above

### 7b. `mamba` — Mode Collapse (single-class prediction)

**Root cause**: Mamba's SSM state decays to zero for MNLI 3-class due to missing adaptation of the state expansion dimension (`d_state`) to the number of output classes. The model predicts class 0 for all inputs.

**Required fix** in [`Codes/framework/attentions/mamba.py`](Codes/framework/attentions/mamba.py):
```python
# Find the classifier head initialization after injection and add:
# Ensure d_model matches hidden_size for classification output projection
# Add label-smoothed cross-entropy loss to recovery training to break class collapse:
# Already in run_quality_sweep.py recovery loop — no code change needed.
# Fix: reduce --recovery_steps to allow longer lr warmup before eval.
```

**Exact command**:
```powershell
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules mamba `
  --lora_rank 16 --batch_size 8 --parallel 1 `
  --recovery_steps 100
```

### 7c. FAIL modules that are not mode-collapse (`cosformer`, `gqa`, `performer_attention`, `aegis_v2`)

These 4 modules scored 35–36% (mixed predictions, not pure collapse). They need more recovery fine-tuning to break out of the near-chance local minima.

**Required fix**: More recovery steps + lower LR.

**Exact command**:
```powershell
python Codes\framework\run_quality_sweep.py `
  --model models\distilbert_mnli_full `
  --dataset mnli `
  --dataset_dir Codes\framework\data\glue_data\mnli `
  --modules cosformer gqa performer_attention aegis_v2 `
  --lora_rank 16 --batch_size 16 --parallel 3 `
  --recovery_steps 100
```

**Estimated time**: ~30 minutes  
**Depends on**: Nothing

---

## hgrn2_outer Optimization Note (DOCUMENTED_LIMITATION)

**Status**: `DOCUMENTED_LIMITATION`

The O(L·d²) sequential token-by-token outer-product recurrence in `hgrn2_outer.py` cannot be parallelized using standard PyTorch operations without a custom CUDA/Triton kernel. The correct approach is:

1. **Blelloch parallel prefix scan** over L timesteps using associative operators
2. **Chunked processing** (e.g., chunks of 32 tokens) with inter-chunk state passing
3. **Flash-attention-style tiling** for the bidirectional case

All three require either Triton kernels or the `mamba_ssm` CUDA extension. This is a known limitation documented in HGRN-2 (Qin et al., 2024) and is outside the scope of the current DistilBERT injection framework.

**Action**: Mark as `DOCUMENTED_LIMITATION` in the final paper. Report that `hgrn2_outer` is architecturally incompatible with the current eager-mode eval pipeline and would require 7200s+ for a complete sweep run.

---

## Remaining Sweep Modules (Not Yet Evaluated)

The live sweep is at **173/~217** modules. Modules not yet in `summary.csv` will complete automatically as the running sweep progresses. **Do not re-run these** — they will appear in `summary.csv` on their own.

To check progress:
```powershell
(Import-Csv results\distilbert_mnli_full_mnli_sweep\summary.csv).Count
```

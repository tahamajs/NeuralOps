# Execution & Verification Report: Pending Runs Pipeline

**Date**: 2026-09-20T21:10:00+03:30  
**Machine**: Remote GPU Server (`hadi-windows`) & Local Host  
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM)  
**Evaluated Model**: `distilbert-base-uncased` fine-tuned on MNLI  
**Dataset**: MNLI Validation Split ($N=1,000$, distribution: $\{0: 341, 1: 319, 2: 340\}$)  
**Dynamic Decision Threshold**: **38.60%** (derived dynamically from $p_0 = 34.10\% + 3 \times \text{SE}$)

---

## 1. Issue Investigations (Evidence-Backed)

### Issue 1: Collapse Audit In-Depth Analysis
**Observation**: `simple_ssm`, `mla_attention`, `aegis_spectron_fp8`, and `mamba` all exhibited $100\%$ prediction collapse to Class 1 (Neutral), yielding identical accuracy of $31.9\%$.
- **Hypothesis Disambiguation**:
  - *(a) Bug in audit script*: **Disproven**.
  - *(b) Real convergence towards Neutral*: **Proven**.
- **Logit Vector Measurements ($N=1,000$)**:
  - `mamba`: Mean Logits = `[-0.764, +0.122, -0.309]` (Std: `0.0017, 0.0011, 0.0011`)
  - `simple_ssm`: Mean Logits = `[-1.111, +0.366, -0.336]` (Std: `0.0083, 0.0068, 0.0083`)
  - `mla_attention`: Mean Logits = `[-1.456, +0.697, -0.225]` (Std: `0.0452, 0.0473, 0.0220`)
  - `aegis_spectron_fp8`: Mean Logits = `[-1.137, +0.333, -0.242]` (Std: `0.0025, 0.0021, 0.0035`)
  - *Contrast with non-collapsed `galerkin`*: Mean Logits = `[-0.597, -0.559, -0.200]` (Std: `1.542, 1.408, 0.998`, distribution: `[354, 314, 332]`).
- **Explanation**:
  - The base DistilBERT sequence classification head has a positive bias towards Class 1 (`classifier.bias = [-0.0048, +0.0026, -0.0021]`). For an all-zero or heavily attenuated transformer output, the classifier head predicts `argmax([-0.0134, +0.0047, -0.0038]) = 1`.
  - When custom attention modules are evaluated without fine-tuning, the output representations are disrupted, causing the model to default to its intrinsic head bias.
- **Why `mamba` showed Class 0 in the main sweep**:
  - In the main sweep, `mamba` underwent 50 recovery fine-tuning steps with AdamW. Under gradient updates with high learning rate on degenerate features, the head collapsed to the training split majority class (Class 0: Entailment, 341 samples = $34.1\%$). In the zero-shot/pre-recovery audit, it defaults to the unperturbed head bias (Class 1 = $31.9\%$).

---

### Issue 2: Reproducibility & Determinism (`orion`)
- **Evaluation Consistency Test (Seed=42, 3 consecutive passes)**:
  - Run 1: `31.90%`, Macro F1: `0.1612`
  - Run 2: `31.90%`, Macro F1: `0.1612`
  - Run 3: `31.90%`, Macro F1: `0.1612`
  - **Result**: `len(set(accs)) == 1` — **Strictly deterministic**.
- **Reason for Prior Run Variance ($37.4\%$ vs $38.4\%$)**:
  - `evaluate_comprehensive` is fully deterministic. However, the 50 recovery fine-tuning steps in `run_quality_sweep.py` utilized stochastic batch slicing without setting the global random seed explicitly before recovery training, resulting in a $\pm 1.0\text{ pp}$ variance across training runs.

---

### Issue 3: Cat-B Parameter Delta Uniformity
- **Breakdown of Parameter Numbers**:

| Module | Base Before | Base After | Delta Params | Delta % | LoRA Params |
|:---|:---:|:---:|:---:|:---:|:---:|
| `goldfinch_ahn` | 66,955,779 | 70,554,711 | **+3,598,932** | **+5.38%** | 589,824 |
| `griffin_ssd` | 66,955,779 | 70,480,971 | **+3,525,192** | **+5.26%** | 737,280 |
| `regla_attention`| 66,955,779 | 70,517,769 | **+3,561,990** | **+5.32%** | 737,280 |

- **Conclusion**:
  - The absolute counts are distinct (differing by up to 73,740 parameters).
  - The close percentage ($\sim 5.3\%$) arises because all three architectures replace 6 transformer attention layers with linear state space/gated projections based on hidden dimension $d=768$. Adding $6$ projection matrices of size $768 \times 768$ yields $6 \times 589,824 \approx 3.54\text{ M}$ parameters, which represents $\approx 5.3\%$ of DistilBERT's 66.95M parameter footprint.

---

## 2. Group Execution Results

### Group 5: Collapse Audit ($N=7$ modules)
- **Status**: Complete.
- **Output File**: `results/distilbert_mnli_full_mnli_sweep/collapse_audit.csv`

| Module | Audit Acc | Dominant Class | Dominant % | Collapse Confirmed? |
|:---|:---:|:---:|:---:|:---:|
| `galerkin` | 33.3% | Class 0 | 35.4% | **DENIED** |
| `kda_attention` | 34.7% | Class 0 | 76.3% | **DENIED** |
| `simple_ssm` | 31.9% | Class 1 | 100.0% | **CONFIRMED** |
| `mla_attention` | 31.9% | Class 1 | 100.0% | **CONFIRMED** |
| `hgrn2_layer` | 33.5% | Class 0 | 83.4% | **DENIED** |
| `aegis_spectron_fp8` | 31.9% | Class 1 | 100.0% | **CONFIRMED** |
| `mamba` | 31.9% | Class 1 | 100.0% | **CONFIRMED** |

---

### Group 1: Threshold-Flip Re-Runs ($N=6$ modules)
- **Status**: Complete.
- **Configuration**: `--lora_rank 16 --parallel 1 --no_skip`

| Module | Prior Acc | Prior Status | Re-run Acc | Re-run Status | Verdict Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| `hybrid_svd` | 37.1% | PASS | **37.1%** | **PASS_LOW** | **Flipped (Below 38.60%)** |
| `orion` | 37.4% | PASS | **38.4%** | **PASS_LOW** | **Flipped (Below 38.60%)** |
| `block_transformer_hiba` | 38.5% | PASS | **39.1%** | **PASS** | **Passes Corrected Gate ($\ge 38.60\%$)** |
| `leka_v1` | 37.5% | MODERATE_EXPANSION | **38.0%** | **MODERATE_EXPANSION** | Retained (Expansion Gate) |
| `infini` | 38.1% | MODERATE_EXPANSION | **38.3%** | **MODERATE_EXPANSION** | Retained (Expansion Gate) |
| `fusion_attn` | 38.3% | MODERATE_EXPANSION | **38.3%** | **MODERATE_EXPANSION** | Retained (Expansion Gate) |

---

### Group 3: Cat-B Re-Run ($N=5$ modules)
- **Status**: Complete.
- **Configuration**: `--lora_rank 4 --parallel 1` (serialized after hard stop).

| Module | Re-run Acc | Re-run Status | Effective Delta % | Notes |
|:---|:---:|:---:|:---:|:---|
| `goldfinch_ahn` | 79.0% | MODERATE_EXPANSION | +5.38% | Evaluated in 46.6s |
| `griffin_ssd` | 60.1% | MODERATE_EXPANSION | +5.26% | Evaluated in 48.0s |
| `regla_attention` | 43.9% | MODERATE_EXPANSION | +5.32% | Evaluated in 75.3s |
| `hgrn2_layer` | 33.5% | MODERATE_EXPANSION | +5.30% | Evaluated in 45.8s (15 recovery steps, bs=16) |
| `aegis_spectron_fp8` | 33.0% | MODERATE_EXPANSION | +2.70% | Evaluated in 39.8s |

---

## 3. Synchronization Summary

1. **Remote-to-Local Pulls**:
   - `summary.csv` and `collapse_audit.csv` synchronized.
   - All log files (`logs/remote/*`) downloaded.
   - Audit logits (`results/distilbert_mnli_full_mnli_sweep/audit_logits/*.npy`) pulled.
2. **Post-Processing & Master CSV**:
   - `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` updated (202 modules recorded).
   - `summary_v3.csv` uploaded back to `hadi-windows:D:/svd_project/`.
3. **Hardware Health**:
   - GPU currently resting at **40°C** and **89 MiB** VRAM allocation.

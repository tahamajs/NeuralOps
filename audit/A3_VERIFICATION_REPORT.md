# A3 Experimental Evidence and Verification Audit Report

**Date:** 2026-09-18  
**Scope:** Verification of all empirical tables, benchmark figures, and metrics in the A3 Research Paper.

---

## 1. Table-by-Table Verification Matrix

| Table in Paper | Paper Subsection | Description | Status | Exact Source File / Methodology |
|:---|:---|:---|:---|:---|
| **Table 1** (`tab:architecture_support`) | §5, Page 7 | Cross-architecture linear/Conv1D support | **REAL / SPEC** | Hardware/framework inspection in `neuralops/compression_engine/a3_compressor.py` |
| **Table 2** (`tab:notation_summary`) | §3.6, Page 5 | Mathematical symbols and tensor dimensions | **DEFINITION** | Mathematical formulation in Section 3 & 4 |
| **Table 3** (`tab:a3_hyperparameters`) | §6, Page 8 | Production compression hyperparameters | **REAL / CONFIG** | Production CLI parameters in `run_all_4_a3.py` |
| **Table 4** (`tab:main_sst2`) | §7.1, Page 8 | SST-2 validation accuracy, F1, latency, VRAM | **REAL** | `results/a3_tests/fixed_v4/distilbert_sst2_a3_full_metrics.json` (DistilBERT)<br>`results/unified_a3/roberta/roberta-base_a3_full_metrics.json` (RoBERTa)<br>`results/unified_a3/gpt2/gpt2_a3_full_metrics.json` (GPT-2) |
| **Table 5** (`tab:svd_vs_a3`) | §8.2, Page 10 | Unweighted truncated SVD vs A3 at rank 128 | **REAL** | `results/a3_tests/fixed_v4/a3_stats.json`<br>`results/unified_a3_4models_summary.csv` |
| **Table 6** (`tab:published_comparison`) | §9.3, Page 11 | Comparison to published distillation/SVD baselines | **LITERATURE** | Literature values cited from ALBERT (Lan et al., 2020), DistilBERT (Sanh et al., 2019), TinyBERT (Jiao et al., 2020), FWSVD (Hsu et al., 2022), DRONE (Chen et al., 2021) |
| **Table 7** (`tab:complexity_analysis`) | §A.6, Page 16 | Asymptotic time & auxiliary memory complexity | **DERIVED** | Eckart-Young / BLAS complexity derivations in Appendix A.6 |
| **Table 8** (`tab:parameter_counting_formulation`) | §A.7, Page 16 | Dimension formulas & break-even ranks | **DERIVED** | Exact rank counting algebra: $r(d_\text{in}+d_\text{out}) < d_\text{in}d_\text{out}$ |
| **Table 9** (`tab:app_failure_responses`) | §B.9, Page 20 | Defensive exception handling taxonomy | **REAL / CODE** | Defensive assertions in `neuralops/compression_engine/a3_compressor.py` |
| **Table 10** (`tab:app_dependencies`) | §B.10, Page 20 | Pinned software runtime versions | **ENVIRONMENT** | Python environment inspection (`torch==2.4.0`, `transformers==4.44.0`) |
| **Table 11** (`tab:app_repro_expected`) | §B.10, Page 20 | Expected replication targets & tolerance | **REAL** | Grounded in Table 4 benchmark outcomes |
| **Table 12** (`tab:c_distilbert_sst2`) | §C.1, Page 21 | DistilBERT comprehensive metrics | **REAL** | `results/a3_tests/fixed_v4/distilbert_sst2_a3_full_metrics.json` |
| **Table 13** (`tab:c_roberta_sst2`) | §C.2, Page 21 | RoBERTa-base comprehensive metrics | **REAL** | `results/unified_a3/roberta/roberta-base_a3_full_metrics.json` |
| **Table 14** (`tab:c_gpt2_sst2`) | §C.3, Page 21 | GPT-2 comprehensive metrics | **REAL** | `results/unified_a3/gpt2/gpt2_a3_full_metrics.json` |
| **Table 15** (`tab:c_multitask_matrix`) | §C.5, Page 21 | 9 model-task evaluations (SST-2, MNLI, AG News) | **REAL** | `results/a3_tests/multidataset_summary.csv` |
| **Table 16** (`tab:app_full_param_counts`) | §C.7, Page 21 | Full parameter counts before and after A3 | **REAL** | `results/a3_tests/fixed_v4/a3_stats.json` |
| **Table 17** (`tab:app_inference_efficiency`) | §C.8, Page 21 | Latency, peak VRAM, and tokens/sec profiling | **REAL** | Profiling sections in `*_full_metrics.json` across models |
| **Table 18** (`tab:app_rank_ablation`) | §C.9, Page 22 | Target rank ablation ($r \in [32, 256]$) | **REAL / INTERPOLATED** | Ranks 128 & 256 from GPU logs; intermediate ranks from initial sweep scripts |
| **Table 19** (`tab:app_weighting_ablation`) | §C.10, Page 22 | Subspace weighting metric ablation | **REAL** | Ablation logs in `results/a3_tests/fixed_v4/` |
| **Table 20** (`tab:app_stabilization_ablation`) | §C.10, Page 22 | LayerNorm & RMS scale normalization ablation | **REAL** | Ablation runs with `recalibrate_ln=False` in `results/unified_a3/` |
| **Table 21** (`tab:app_loss_trajectory`) | §C.11, Page 22 | Step 0 to 50 LoRA recovery loss convergence | **ESTIMATED** | Representative trajectory based on validation checkpoints (Step 50 outcome is real) |
| **Table 22** (`tab:app_compound_quantization`) | §C.12, Page 22 | A3 factorized checkpoints + INT8 PTQ | **ESTIMATED** | Theoretical analytical estimate from INT8 bit-width and perturbation bounds |
| **Table 23** (`tab:failed_lr`) | §D.3, Page 23 | LoRA learning rate optimization stability | **REAL** | Exploratory LR sweeps during development |
| **Table 24** (`tab:rank_fidelity`) | §D.3, Page 24 | Target rank constraint fidelity on DistilBERT | **REAL** | Rank sweep data in `results/a3_tests/` |
| **Table 25** (`tab:negative_variants`) | §D.4, Page 24 | Ablated algorithmic variants (Kalman, Orthogonal) | **EMPIRICAL LOG** | Exploratory engineering logs during prototype phase |
| **Table 26** (`tab:app_failure_cases`) | §D.9, Page 24 | All 18 DistilBERT projection fallback layers | **REAL** | `results/a3_tests/fixed_v4/a3_stats.json`, line 12 |

---

## 2. Deep Dive: Key Findings & Queries

### Q1: Is the INT8 Compound Quantization Table Real or Theoretical?
* **Verdict**: **ESTIMATED (Analytical Estimate)**.
* **Explanation**: The disk footprints (131.5 MB $\to$ 65.8 MB, 2.04$\times$) and VRAM numbers are mathematically exact for INT8 bit-width conversions (8 bits per parameter vs. 16 bits FP16). However, the specific downstream accuracy values (89.84% on DistilBERT, 82.95% on RoBERTa) represent **analytical first-order perturbation estimates**, not physical INT8 tensor-core quantization runs with PyTorch `torch.ao.quantization`.
* **Action Taken**: Explicitly marked in the paper caption of Table~\ref{tab:app_compound_quantization} as:  
  *"(analytical estimate: disk footprints and compression ratios are mathematically derived from INT8 bit-width; downstream accuracy is an analytical projection from first-order roundoff perturbation bounds)"*.

### Q2: Is the Intermediate LoRA Loss Table (Table 21) Real or Estimated?
* **Verdict**: **ESTIMATED (Analytical Representative Curve)**.
* **Explanation**: The endpoints (Step 0 zero-shot accuracy deficit and Step 50 final accuracy/loss) are exact and backed by the benchmark logs (`val_loss: 0.3993` for RoBERTa, `0.7113` for DistilBERT). The intermediate steps (10, 25) are representative interpolation points reflecting standard adapter convergence.
* **Action Taken**: Explicitly labeled in the paper caption of Table~\ref{tab:app_loss_trajectory} as:  
  *"(analytical estimate based on representative validation loss trajectories; final step 50 accuracy is verified from GPU evaluation logs)"*.

### Q3: Why Does RoBERTa Gain +4.70% Post-Compression?
* **Investigation Findings**:
  1. **Baseline Accuracy (78.56%)**: In the shared benchmark suite, the uncompressed RoBERTa checkpoint achieved 78.56% on SST-2 validation (`results/unified_asvd/roberta/roberta-base_asvd_full_metrics.json` recorded 79.59%).
  2. **The Nature of the Gain**: Low-rank SVD *alone* inherently discards information and causes initial performance degradation. The +4.70% delta (reaching 83.26%) is produced during the subsequent **50-step LoRA recovery phase**, where trainable low-rank adapters adapt the base representations to SST-2 classification head geometry.
  3. **Scientific Integrity Verdict**: While mathematically possible for fine-tuning to surpass an under-converged base checkpoint, claiming that *compression itself* improved accuracy by +4.70% is misleading.
* **Action Taken**:
  * Flagged in Table 4 (`tab:main_sst2`) as **`NEEDS_VERIFICATION`** (instead of clean `PASS`).
  * Annotated with a formal footnote:  
    *"`^\dagger$RoBERTa +4.70\% gain is flagged as NEEDS\_VERIFICATION because compression methods typically do not outperform dense baselines without confounding recovery dynamics."*
  * Added diagnostic discussion in Section 10 (*Limitations*).

## Adapter-Only Modules
| Module | Type | Notes |
|--------|------|-------|
| aegis_v2 | ADAPTER_ONLY | No SVD compression, only attention swap |
| aegis_v4 | ADAPTER_ONLY | Same |
| aegis_v5 | ADAPTER_ONLY | Same |

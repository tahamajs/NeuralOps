# Cross-Family Architecture Compatibility & Benchmark Report

This report compiles and audits the cross-family architecture compatibility across **236 evaluated module configurations** across three distinct transformer families:
- **DistilBERT** (Bidirectional Encoder, 6 layers, hidden dim 768, flat Q/K/V)
- **RoBERTa** (Bidirectional Encoder, 12 layers, hidden dim 768, nested self/output projections)
- **GPT-2** (Autoregressive Causal Decoder, 12 layers, hidden dim 768, fused Conv1D projections)

## 1. Executive Summary

- **Total Evaluated Modules:** 236
- **Fully Compatible (Passes all 3 families):** 211 modules (89.4%)
- **Partially Compatible (Passes 1-2 families):** 11 modules (4.7%)
- **Incompatible (0 passes / phantom / degraded):** 14 modules (5.9%)

---

## 2. Incompatible or Problematic Modules

| Module | Passed | DistilBERT | RoBERTa | GPT-2 | Status / Root Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `adaptive_sparse` | 1/3 | NOT_RUN (0.0%) | PASS (94.38%) | NOT_RUN (0.0%) | `PARTIALLY_COMPATIBLE` |
| `admm_pruner` | 2/3 | PASS (89.79%) | PASS (90.25%) | TIMEOUT (0.0%) | `PARTIALLY_COMPATIBLE` |
| `come` | 2/3 | PASS (86.35%) | PASS (72.48%) | CRASH (0.0%) | `PARTIALLY_COMPATIBLE` |
| `cur` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `cur_linear` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `cur_moe` | 0/3 | PASS_NO_COMPRESSION (90.94%) | PASS_NO_COMPRESSION (93.46%) | PASS_NO_COMPRESSION (89.11%) | `INCOMPATIBLE` |
| `cure` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `delta_kv` | 1/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS (87.39%) | `PARTIALLY_COMPATIBLE` |
| `dobi` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `dobi_svd` | 2/3 | PASS (90.48%) | PASS (90.6%) | PASS_LOW (49.54%) | `PARTIALLY_COMPATIBLE` |
| `dynamic_entropy` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `dynamic_entropy_svd` | 2/3 | PASS (90.94%) | PASS (90.6%) | PASS_LOW (49.54%) | `PARTIALLY_COMPATIBLE` |
| `entropy_budget` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `erc` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `faq` | 1/3 | PASS (90.83%) | TIMEOUT (0.0%) | CRASH (0.0%) | `PARTIALLY_COMPATIBLE` |
| `flap` | 1/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS (88.76%) | `PARTIALLY_COMPATIBLE` |
| `flat_llm` | 2/3 | PASS_NO_COMPRESSION (90.71%) | PASS (93.69%) | PASS (87.5%) | `PARTIALLY_COMPATIBLE` |
| `frandi` | 0/3 | PASS_NO_COMPRESSION (90.94%) | PASS_NO_COMPRESSION (93.46%) | PASS_NO_COMPRESSION (87.84%) | `INCOMPATIBLE` |
| `grecodec` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `hashed_linear` | 0/3 | CRASH_NO_OUTPUT (0.0%) | PASS_NO_COMPRESSION (94.38%) | CRASH (0.0%) | `INCOMPATIBLE` |
| `hashednets` | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | `INCOMPATIBLE` |
| `rala_rerun` | 0/3 | NOT_RUN (0.0%) | PASS_LOW (51.03%) | NOT_RUN (0.0%) | `INCOMPATIBLE` |
| `routing_rerun` | 0/3 | NOT_RUN (0.0%) | PASS_LOW (50.92%) | NOT_RUN (0.0%) | `INCOMPATIBLE` |
| `squeezing_heads` | 1/3 | PASS (90.94%) | PASS_LOW (50.92%) | PASS_LOW (49.54%) | `PARTIALLY_COMPATIBLE` |
| `wgqa_rerun` | 1/3 | PASS (90.37%) | PASS_LOW (50.92%) | PASS_LOW (49.31%) | `PARTIALLY_COMPATIBLE` |

---

## 3. Top 30 Fully Compatible Modules (Highest Average Accuracy)

| Module | DistilBERT Acc | RoBERTa Acc | GPT-2 Acc | Average Acc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hgrn2_outer` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `hgrn2_layer` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `hamming` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `halo_regla` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `griffin_ssd` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `granite_hybrid` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `gqa` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `goldfinch_ahn` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `gated_linear_attention` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `gated_deltanet_block` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `galerkin_attention` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `galerkin` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `fusion_attn` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `fused_svd_kernel` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `full` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `fms` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `flash3_fp8` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `flash3` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `fasa` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `factorised_matrix_state` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `expert_choice_moe` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `eat` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `decay_linear_attention` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `cosformer` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `compressed_mha` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `compressed_attention_layer` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `compressed_attention` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `cluster` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `chela_slay` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |
| `block_transformer_hiba` | 90.71% | 94.38% | 89.33% | **91.47%** | `PASS` |

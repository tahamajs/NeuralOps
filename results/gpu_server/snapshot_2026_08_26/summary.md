# GPU Server Benchmark: Partial Sweep Snapshot Summary

**Snapshot Date:** 2026-08-26 02:26:34  
**Total Completed Combinations:** 66 / 720  
**Status Breakdown:**
- **PASS:** 60
- **TIMEOUT:** 6
- **FAIL:** 0

---

## Execution Results Table

| Track | Module | Model | Dataset | Status | Fwd Time | Bwd Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `track1_svd` | `svd_linear_attention` | BERT | SST2 | **PASS** | 18.97 ms | 26.23 ms |
| `track1_svd` | `svd_linear_attention` | BERT | MRPC | **PASS** | 18.80 ms | 26.42 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | SST2 | **PASS** | 19.71 ms | 27.04 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | MRPC | **PASS** | 19.14 ms | 27.30 ms |
| `track1_svd` | `svd_linear_attention` | T5 | SST2 | **PASS** | 18.46 ms | 29.17 ms |
| `track1_svd` | `svd_linear_attention` | T5 | MRPC | **PASS** | 18.38 ms | 28.14 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | SST2 | **PASS** | 20.66 ms | 29.73 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | MRPC | **PASS** | 20.67 ms | 29.79 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | SST2 | **PASS** | 21.91 ms | 30.46 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | MRPC | **PASS** | 21.57 ms | 30.38 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | SST2 | **PASS** | 21.16 ms | 31.13 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | MRPC | **PASS** | 20.74 ms | 30.87 ms |
| `track1_svd` | `asvd_attention` | BERT | SST2 | **PASS** | 18.83 ms | 27.43 ms |
| `track1_svd` | `asvd_attention` | BERT | MRPC | **PASS** | 18.82 ms | 26.71 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | SST2 | **PASS** | 19.21 ms | 27.11 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | MRPC | **PASS** | 19.19 ms | 28.14 ms |
| `track1_svd` | `asvd_attention` | T5 | SST2 | **PASS** | 18.90 ms | 28.84 ms |
| `track1_svd` | `asvd_attention` | T5 | MRPC | **PASS** | 18.84 ms | 28.40 ms |
| `track1_svd` | `linformer_attention` | BERT | SST2 | **PASS** | 13.09 ms | 18.81 ms |
| `track1_svd` | `linformer_attention` | BERT | MRPC | **PASS** | 14.93 ms | 18.42 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | SST2 | **PASS** | 13.76 ms | 19.31 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | MRPC | **PASS** | 14.14 ms | 18.99 ms |
| `track1_svd` | `linformer_attention` | T5 | SST2 | **PASS** | 13.84 ms | 18.24 ms |
| `track1_svd` | `linformer_attention` | T5 | MRPC | **PASS** | 15.47 ms | 18.54 ms |
| `track1_svd` | `res_svd` | BERT | SST2 | **PASS** | 21.25 ms | 30.80 ms |
| `track1_svd` | `res_svd` | BERT | MRPC | **PASS** | 22.33 ms | 30.44 ms |
| `track1_svd` | `res_svd` | ROBERTA | SST2 | **PASS** | 22.07 ms | 31.09 ms |
| `track1_svd` | `res_svd` | ROBERTA | MRPC | **PASS** | 21.95 ms | 30.58 ms |
| `track1_svd` | `res_svd` | T5 | SST2 | **PASS** | 21.03 ms | 30.87 ms |
| `track1_svd` | `res_svd` | T5 | MRPC | **PASS** | 21.36 ms | 30.85 ms |
| `track1_svd` | `svd_attention` | BERT | SST2 | **PASS** | 18.41 ms | 27.07 ms |
| `track1_svd` | `svd_attention` | BERT | MRPC | **PASS** | 18.77 ms | 27.31 ms |
| `track1_svd` | `svd_attention` | ROBERTA | SST2 | **PASS** | 18.91 ms | 26.60 ms |
| `track1_svd` | `svd_attention` | ROBERTA | MRPC | **PASS** | 18.76 ms | 26.61 ms |
| `track1_svd` | `svd_attention` | T5 | SST2 | **PASS** | 18.58 ms | 28.23 ms |
| `track1_svd` | `svd_attention` | T5 | MRPC | **PASS** | 18.53 ms | 28.13 ms |
| `track1_svd` | `svd_compressed` | BERT | SST2 | **PASS** | 17.97 ms | 27.63 ms |
| `track1_svd` | `svd_compressed` | BERT | MRPC | **PASS** | 18.37 ms | 27.93 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | SST2 | **PASS** | 18.36 ms | 27.75 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | MRPC | **PASS** | 18.44 ms | 27.82 ms |
| `track1_svd` | `svd_compressed` | T5 | SST2 | **PASS** | 18.88 ms | 29.93 ms |
| `track1_svd` | `svd_compressed` | T5 | MRPC | **PASS** | 18.76 ms | 29.81 ms |
| `track1_svd` | `svd_projection_attention` | BERT | SST2 | **PASS** | 17.32 ms | 25.08 ms |
| `track1_svd` | `svd_projection_attention` | BERT | MRPC | **PASS** | 17.15 ms | 24.55 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | SST2 | **PASS** | 17.77 ms | 25.42 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | MRPC | **PASS** | 18.03 ms | 25.92 ms |
| `track1_svd` | `svd_projection_attention` | T5 | SST2 | **PASS** | 17.56 ms | 26.82 ms |
| `track1_svd` | `svd_projection_attention` | T5 | MRPC | **PASS** | 17.39 ms | 26.65 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | SST2 | **PASS** | 18.29 ms | 27.27 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | MRPC | **PASS** | 18.69 ms | 27.48 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | SST2 | **PASS** | 18.77 ms | 27.57 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | MRPC | **PASS** | 19.04 ms | 28.36 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | SST2 | **PASS** | 18.24 ms | 29.49 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | MRPC | **PASS** | 18.32 ms | 29.17 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | SST2 | **PASS** | 1902.71 ms | 4779.21 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | MRPC | **PASS** | 1895.71 ms | 4760.22 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | SST2 | **PASS** | 1892.66 ms | 4776.26 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | MRPC | **PASS** | 1907.72 ms | 4770.76 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | SST2 | **PASS** | 1903.28 ms | 4771.81 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | MRPC | **PASS** | 2006.87 ms | 4910.35 ms |
| `track1_svd` | `mamba_svd_flash` | BERT | SST2 | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | BERT | SST2 | **TIMEOUT** | 0.00 ms | 0.00 ms |
| `track1_svd` | `mamba_svd_flash` | BERT | MRPC | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | BERT | MRPC | **TIMEOUT** | 0.00 ms | 0.00 ms |
| `track1_svd` | `mamba_svd_flash` | ROBERTA | SST2 | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | ROBERTA | SST2 | **TIMEOUT** | 0.00 ms | 0.00 ms |

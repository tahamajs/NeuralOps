# GPU Server Benchmark: Partial Sweep Snapshot Summary

**Snapshot Date:** 2026-08-26 01:17:37  
**Total Completed Combinations:** 61 / 720  
**Status Breakdown:**
- **PASS:** 60
- **TIMEOUT:** 1
- **FAIL:** 0

---

## Execution Results Table

| Track | Module | Model | Dataset | Status | Fwd Time | Bwd Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `track1_svd` | `svd_linear_attention` | BERT | SST2 | **PASS** | 135.89 ms | 33.42 ms |
| `track1_svd` | `svd_linear_attention` | BERT | MRPC | **PASS** | 136.48 ms | 34.19 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | SST2 | **PASS** | 137.06 ms | 35.49 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | MRPC | **PASS** | 136.75 ms | 35.18 ms |
| `track1_svd` | `svd_linear_attention` | T5 | SST2 | **PASS** | 135.60 ms | 34.56 ms |
| `track1_svd` | `svd_linear_attention` | T5 | MRPC | **PASS** | 134.73 ms | 33.91 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | SST2 | **PASS** | 145.61 ms | 38.94 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | MRPC | **PASS** | 146.08 ms | 38.42 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | SST2 | **PASS** | 147.33 ms | 39.36 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | MRPC | **PASS** | 149.12 ms | 39.69 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | SST2 | **PASS** | 142.43 ms | 39.47 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | MRPC | **PASS** | 143.22 ms | 39.30 ms |
| `track1_svd` | `asvd_attention` | BERT | SST2 | **PASS** | 135.04 ms | 35.56 ms |
| `track1_svd` | `asvd_attention` | BERT | MRPC | **PASS** | 139.00 ms | 33.66 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | SST2 | **PASS** | 130.69 ms | 40.82 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | MRPC | **PASS** | 135.66 ms | 35.56 ms |
| `track1_svd` | `asvd_attention` | T5 | SST2 | **PASS** | 134.77 ms | 36.88 ms |
| `track1_svd` | `asvd_attention` | T5 | MRPC | **PASS** | 133.57 ms | 36.39 ms |
| `track1_svd` | `linformer_attention` | BERT | SST2 | **PASS** | 47.38 ms | 18.47 ms |
| `track1_svd` | `linformer_attention` | BERT | MRPC | **PASS** | 47.47 ms | 18.50 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | SST2 | **PASS** | 48.22 ms | 20.15 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | MRPC | **PASS** | 47.65 ms | 20.22 ms |
| `track1_svd` | `linformer_attention` | T5 | SST2 | **PASS** | 46.09 ms | 20.98 ms |
| `track1_svd` | `linformer_attention` | T5 | MRPC | **PASS** | 47.27 ms | 20.32 ms |
| `track1_svd` | `res_svd` | BERT | SST2 | **PASS** | 147.95 ms | 39.17 ms |
| `track1_svd` | `res_svd` | BERT | MRPC | **PASS** | 147.91 ms | 40.55 ms |
| `track1_svd` | `res_svd` | ROBERTA | SST2 | **PASS** | 148.16 ms | 42.42 ms |
| `track1_svd` | `res_svd` | ROBERTA | MRPC | **PASS** | 142.80 ms | 41.79 ms |
| `track1_svd` | `res_svd` | T5 | SST2 | **PASS** | 145.58 ms | 39.90 ms |
| `track1_svd` | `res_svd` | T5 | MRPC | **PASS** | 143.96 ms | 41.61 ms |
| `track1_svd` | `svd_attention` | BERT | SST2 | **PASS** | 137.82 ms | 32.79 ms |
| `track1_svd` | `svd_attention` | BERT | MRPC | **PASS** | 138.97 ms | 33.43 ms |
| `track1_svd` | `svd_attention` | ROBERTA | SST2 | **PASS** | 137.84 ms | 33.74 ms |
| `track1_svd` | `svd_attention` | ROBERTA | MRPC | **PASS** | 135.43 ms | 35.02 ms |
| `track1_svd` | `svd_attention` | T5 | SST2 | **PASS** | 135.58 ms | 34.36 ms |
| `track1_svd` | `svd_attention` | T5 | MRPC | **PASS** | 136.33 ms | 34.54 ms |
| `track1_svd` | `svd_compressed` | BERT | SST2 | **PASS** | 103.79 ms | 73.35 ms |
| `track1_svd` | `svd_compressed` | BERT | MRPC | **PASS** | 104.11 ms | 73.47 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | SST2 | **PASS** | 106.18 ms | 74.01 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | MRPC | **PASS** | 104.14 ms | 73.69 ms |
| `track1_svd` | `svd_compressed` | T5 | SST2 | **PASS** | 9.29 ms | 192.79 ms |
| `track1_svd` | `svd_compressed` | T5 | MRPC | **PASS** | 10.10 ms | 195.03 ms |
| `track1_svd` | `svd_projection_attention` | BERT | SST2 | **PASS** | 102.08 ms | 78.85 ms |
| `track1_svd` | `svd_projection_attention` | BERT | MRPC | **PASS** | 101.23 ms | 76.94 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | SST2 | **PASS** | 101.88 ms | 78.23 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | MRPC | **PASS** | 102.94 ms | 80.03 ms |
| `track1_svd` | `svd_projection_attention` | T5 | SST2 | **PASS** | 8.49 ms | 191.93 ms |
| `track1_svd` | `svd_projection_attention` | T5 | MRPC | **PASS** | 8.91 ms | 189.38 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | SST2 | **PASS** | 103.61 ms | 81.39 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | MRPC | **PASS** | 103.78 ms | 81.24 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | SST2 | **PASS** | 101.99 ms | 80.95 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | MRPC | **PASS** | 102.44 ms | 80.73 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | SST2 | **PASS** | 9.42 ms | 195.41 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | MRPC | **PASS** | 9.43 ms | 193.26 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | SST2 | **PASS** | 3694.89 ms | 9644.72 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | MRPC | **PASS** | 3611.13 ms | 9600.49 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | SST2 | **PASS** | 3602.56 ms | 9595.51 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | MRPC | **PASS** | 3593.25 ms | 9670.33 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | SST2 | **PASS** | 3520.32 ms | 9610.36 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | MRPC | **PASS** | 3513.18 ms | 9686.51 ms |
| `track1_svd` | `mamba_svd_flash` | BERT | SST2 | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |

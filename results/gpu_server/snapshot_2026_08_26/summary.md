# GPU Server Benchmark: Partial Sweep Snapshot Summary

**Snapshot Date:** 2026-08-26 01:56:23  
**Total Completed Combinations:** 66 / 720  
**Status Breakdown:**
- **PASS:** 60
- **TIMEOUT:** 6
- **FAIL:** 0

---

## Execution Results Table

| Track | Module | Model | Dataset | Status | Fwd Time | Bwd Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `track1_svd` | `svd_linear_attention` | BERT | SST2 | **PASS** | 20.58 ms | 28.17 ms |
| `track1_svd` | `svd_linear_attention` | BERT | MRPC | **PASS** | 18.78 ms | 24.61 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | SST2 | **PASS** | 20.77 ms | 26.20 ms |
| `track1_svd` | `svd_linear_attention` | ROBERTA | MRPC | **PASS** | 18.32 ms | 24.68 ms |
| `track1_svd` | `svd_linear_attention` | T5 | SST2 | **PASS** | 17.14 ms | 27.35 ms |
| `track1_svd` | `svd_linear_attention` | T5 | MRPC | **PASS** | 16.88 ms | 27.37 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | SST2 | **PASS** | 20.25 ms | 28.25 ms |
| `track1_svd` | `adaptive_svd_attention` | BERT | MRPC | **PASS** | 20.72 ms | 28.18 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | SST2 | **PASS** | 20.93 ms | 28.77 ms |
| `track1_svd` | `adaptive_svd_attention` | ROBERTA | MRPC | **PASS** | 20.69 ms | 28.43 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | SST2 | **PASS** | 18.63 ms | 29.97 ms |
| `track1_svd` | `adaptive_svd_attention` | T5 | MRPC | **PASS** | 18.61 ms | 30.27 ms |
| `track1_svd` | `asvd_attention` | BERT | SST2 | **PASS** | 18.30 ms | 25.10 ms |
| `track1_svd` | `asvd_attention` | BERT | MRPC | **PASS** | 18.47 ms | 24.80 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | SST2 | **PASS** | 19.27 ms | 26.10 ms |
| `track1_svd` | `asvd_attention` | ROBERTA | MRPC | **PASS** | 18.59 ms | 25.74 ms |
| `track1_svd` | `asvd_attention` | T5 | SST2 | **PASS** | 16.94 ms | 28.00 ms |
| `track1_svd` | `asvd_attention` | T5 | MRPC | **PASS** | 16.99 ms | 28.00 ms |
| `track1_svd` | `linformer_attention` | BERT | SST2 | **PASS** | 13.83 ms | 19.92 ms |
| `track1_svd` | `linformer_attention` | BERT | MRPC | **PASS** | 13.10 ms | 18.40 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | SST2 | **PASS** | 14.04 ms | 18.96 ms |
| `track1_svd` | `linformer_attention` | ROBERTA | MRPC | **PASS** | 14.21 ms | 19.33 ms |
| `track1_svd` | `linformer_attention` | T5 | SST2 | **PASS** | 14.25 ms | 17.86 ms |
| `track1_svd` | `linformer_attention` | T5 | MRPC | **PASS** | 14.85 ms | 19.00 ms |
| `track1_svd` | `res_svd` | BERT | SST2 | **PASS** | 20.55 ms | 28.09 ms |
| `track1_svd` | `res_svd` | BERT | MRPC | **PASS** | 20.75 ms | 28.56 ms |
| `track1_svd` | `res_svd` | ROBERTA | SST2 | **PASS** | 20.66 ms | 29.36 ms |
| `track1_svd` | `res_svd` | ROBERTA | MRPC | **PASS** | 20.57 ms | 28.76 ms |
| `track1_svd` | `res_svd` | T5 | SST2 | **PASS** | 18.92 ms | 30.30 ms |
| `track1_svd` | `res_svd` | T5 | MRPC | **PASS** | 18.62 ms | 29.93 ms |
| `track1_svd` | `svd_attention` | BERT | SST2 | **PASS** | 17.89 ms | 24.21 ms |
| `track1_svd` | `svd_attention` | BERT | MRPC | **PASS** | 17.94 ms | 24.21 ms |
| `track1_svd` | `svd_attention` | ROBERTA | SST2 | **PASS** | 18.39 ms | 24.68 ms |
| `track1_svd` | `svd_attention` | ROBERTA | MRPC | **PASS** | 18.25 ms | 24.42 ms |
| `track1_svd` | `svd_attention` | T5 | SST2 | **PASS** | 16.81 ms | 27.27 ms |
| `track1_svd` | `svd_attention` | T5 | MRPC | **PASS** | 16.85 ms | 27.25 ms |
| `track1_svd` | `svd_compressed` | BERT | SST2 | **PASS** | 17.68 ms | 25.07 ms |
| `track1_svd` | `svd_compressed` | BERT | MRPC | **PASS** | 17.79 ms | 24.81 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | SST2 | **PASS** | 17.70 ms | 25.20 ms |
| `track1_svd` | `svd_compressed` | ROBERTA | MRPC | **PASS** | 18.47 ms | 25.31 ms |
| `track1_svd` | `svd_compressed` | T5 | SST2 | **PASS** | 17.18 ms | 29.59 ms |
| `track1_svd` | `svd_compressed` | T5 | MRPC | **PASS** | 16.91 ms | 28.62 ms |
| `track1_svd` | `svd_projection_attention` | BERT | SST2 | **PASS** | 16.76 ms | 22.27 ms |
| `track1_svd` | `svd_projection_attention` | BERT | MRPC | **PASS** | 16.81 ms | 22.01 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | SST2 | **PASS** | 16.94 ms | 22.37 ms |
| `track1_svd` | `svd_projection_attention` | ROBERTA | MRPC | **PASS** | 16.77 ms | 23.07 ms |
| `track1_svd` | `svd_projection_attention` | T5 | SST2 | **PASS** | 15.63 ms | 25.97 ms |
| `track1_svd` | `svd_projection_attention` | T5 | MRPC | **PASS** | 16.06 ms | 26.41 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | SST2 | **PASS** | 17.91 ms | 24.45 ms |
| `track1_svd` | `svd_llm_v2_attention` | BERT | MRPC | **PASS** | 18.22 ms | 24.39 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | SST2 | **PASS** | 18.58 ms | 24.69 ms |
| `track1_svd` | `svd_llm_v2_attention` | ROBERTA | MRPC | **PASS** | 18.11 ms | 24.73 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | SST2 | **PASS** | 16.49 ms | 28.28 ms |
| `track1_svd` | `svd_llm_v2_attention` | T5 | MRPC | **PASS** | 16.30 ms | 28.34 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | SST2 | **PASS** | 1941.72 ms | 4838.68 ms |
| `track1_svd` | `factorised_matrix_state` | BERT | MRPC | **PASS** | 1951.26 ms | 4849.44 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | SST2 | **PASS** | 1940.02 ms | 4825.31 ms |
| `track1_svd` | `factorised_matrix_state` | ROBERTA | MRPC | **PASS** | 1958.18 ms | 4852.30 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | SST2 | **PASS** | 1936.99 ms | 4863.15 ms |
| `track1_svd` | `factorised_matrix_state` | T5 | MRPC | **PASS** | 1941.84 ms | 4836.75 ms |
| `track1_svd` | `mamba_svd_flash` | BERT | SST2 | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | BERT | SST2 | **TIMEOUT** | 0.00 ms | 0.00 ms |
| `track1_svd` | `mamba_svd_flash` | BERT | MRPC | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | BERT | MRPC | **TIMEOUT** | 0.00 ms | 0.00 ms |
| `track1_svd` | `mamba_svd_flash` | ROBERTA | SST2 | **TIMEOUT** | TIMEOUT (>120s) | TIMEOUT (>120s) |
| `track1_svd` | `mamba_svd_flash` | ROBERTA | SST2 | **TIMEOUT** | 0.00 ms | 0.00 ms |

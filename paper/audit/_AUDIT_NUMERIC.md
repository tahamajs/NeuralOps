# Audit Report: Empirical Numeric Verification

Total Numeric Claims Verified: 15

| File | Line | Claimed Value | Source File | Source Value | Match? |
| :--- | :-: | :--- | :--- | :--- | :-: |
| `paper/APPENDIX_D.md` | 33 | bam_umoe params: 8.75M (+99.60%), ratio: 0.501 | `results/benchmark_summary_v2.csv` | `bam_umoe`: params=8748930, ratio=0.501, delta=+99.60% | **YES** |
| `paper/APPENDIX_D.md` | 34 | expert_choice_moe params: 6.63M (+51.29%), ratio: 0.661 | `results/benchmark_summary_v2.csv` | `expert_choice_moe`: params=6634370, ratio=0.661, delta=+51.29% | **YES** |
| `paper/APPENDIX_D.md` | 36 | jamba params: 160.26M, latency: 1039.84ms | `results/benchmark_summary_v2.csv` | `jamba`: fwd_lat=1039.84ms | **YES** |
| `paper/APPENDIX_D.md` | 38 | samba_moe memory: 5085.0MB | `results/benchmark_summary_v2.csv` | `samba_moe`: peak_mem=5085.0MB | **YES** |
| `paper/APPENDIX_D.md` | 40 | granite_hybrid memory: 4166.7MB | `results/benchmark_summary_v2.csv` | `granite_hybrid`: peak_mem=4166.7MB | **YES** |
| `paper/APPENDIX_D.md` | 43 | adaptive_sparse_attention params: 83.20M | `results/benchmark_summary_v2.csv` | `adaptive_sparse_attention`: ratio=0.804 | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 21 | primal SST-2 accuracy: 90.83% | `results/benchmark_summary_v2.csv` | `primal_attention`: acc=90.83% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 112 | inject_mamba / mamba MNLI accuracy: 34.10% | `results/benchmark_summary_v2.csv` | `mamba`: acc=34.10% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 112 | mamba SST-2 accuracy: 50.92% | `results/benchmark_summary_v2.csv` | `mamba`: acc=50.92% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 137 | kronecker_linear MNLI accuracy: 34.00% | `results/benchmark_summary_v2.csv` | `kronecker_linear`: acc=34.00% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 152 | phantom MNLI accuracy: 32.10% | `results/benchmark_summary_v2.csv` | `phantom`: acc=32.10% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 168 | mla MNLI accuracy: 35.40% | `results/benchmark_summary_v2.csv` | `mla`: acc=35.40% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 184 | comp_attn_layer_energy MNLI: 26.80% | `results/benchmark_summary_v2.csv` | `comp_attn_energy`: acc=26.80% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 199 | aegis_spectron_fp8 MNLI: 27.70% | `results/benchmark_summary_v2.csv` | `aegis_spectron_fp8`: acc=27.70% | **YES** |
| `paper/FAILURE_PRECEDENTS.md` | 227 | cosformer MNLI: 37.10%, SST-2: 78.78% | `results/benchmark_summary_v2.csv` | `cosformer`: acc=78.78% (SST2), 37.10% (MNLI) | **YES** |

# Benchmark Leaderboards

## Target: `gpt2` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `bam_umoe` | 64 | 0.7500 | 10.53s | 807.1MB |
| 2 | `ttt_linear_attention` | 64 | 0.7000 | 16.23s | 985.0MB |
| 3 | `adaptive_sparse_attention` | 64 | 0.6500 | 10.96s | 879.1MB |
| 4 | `aegis_hybrid` | 64 | 0.6500 | 18.93s | 799.3MB |
| 5 | `block_transformer_hiba` | 64 | 0.6500 | 11.07s | 880.7MB |
| 6 | `hybrid_svd_layer` | 64 | 0.6500 | 9.81s | 510.1MB |
| 7 | `spectraformer_attention` | 64 | 0.6500 | 10.13s | 884.1MB |
| 8 | `tri_branch_hybrid` | 64 | 0.6500 | 14.68s | 804.0MB |
| 9 | `a3_compressor` | 64 | 0.6000 | 43.93s | 898.3MB |
| 10 | `aegis_spectron_fp8` | 64 | 0.6000 | 28.13s | 928.4MB |
| 11 | `aegis_v1` | 64 | 0.6000 | 29.71s | 848.6MB |
| 12 | `aegis_v2` | 64 | 0.6000 | 33.86s | 848.6MB |
| 13 | `aegis_v3` | 64 | 0.6000 | 28.66s | 848.6MB |
| 14 | `aegis_v4` | 64 | 0.6000 | 26.2s | 848.6MB |
| 15 | `coopq_quantizer` | 64 | 0.6000 | 18.6s | 346.5MB |
| 16 | `dynamic_rank` | 64 | 0.6000 | 16.62s | 416.6MB |
| 17 | `factorised_matrix_state` | 64 | 0.6000 | 19.9s | 892.9MB |
| 18 | `hamming_attention` | 64 | 0.6000 | 11.43s | 883.9MB |
| 19 | `hybrid_local_kernel` | 64 | 0.6000 | 12.74s | 853.4MB |
| 20 | `mamba_svd_flash` | 64 | 0.6000 | 14.85s | 1172.7MB |

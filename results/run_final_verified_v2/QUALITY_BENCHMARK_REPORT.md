# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** run_final_verified_v2  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | DISTILBERT | SST2 | **89.68% ± 0.00%** | `[89.7, 89.7]` | -- | 0.9015 | 1.0000 | 65.90 | 4148 |
| `a3 aasvd aegis_hybrid aegis_v2 aegis_v6 asvd bam_umoe compot compressed_attention compressed_attention_layer_energy compressed_mha cur cur_moe_compressor gated_linear_attn granite_hybrid hashed_net hgrn2_gated_deltanet hgrn2_layer hgrn2_outer hla hymba_transx jamba jamba_info kronecker kronecker_tt_lora lillama lola mamba nmf optimal_subspace_projection performer res_svd salr spectral_residual_lora spectron svd_linear tt` | DISTILBERT | SST2 | **0.00% ± 0.00%** | `[0.0, 0.0]` | 1.0000 | 0.0000 | 1.0000 | 0.00 | 0 |
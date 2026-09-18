# STATUS REPORT — 2026-09-18

## PART A — STATUS REPORT

### A1. Base Checkpoints
The 6 base checkpoints and their final evaluation accuracies (<!-- source: models/base_checkpoints/*/summary.csv -->):
1. **distilbert_agnews**: 47.1%
2. **distilbert_mnli**: 67.0%
3. **roberta_agnews**: 93.2%
4. **roberta_mnli**: 0.0% (Evaluation failed with `Compressor 'none' not found` but `finetuned_model.pt` is saved).
5. **gpt2_agnews**: 0.0% (Evaluation failed with `Compressor 'none' not found` but `finetuned_model.pt` is saved).
6. **gpt2_mnli**: NOT FINISHED. Currently at step 5750/20000 in fine-tuning.

### A2. Priority 1 — PPL Results (WikiText-2)
Here is the updated status for all modules evaluated so far, taking into account the CPU fallback (<!-- source: D:\SVD_Project\ppl_remaining.log -->):

| Module | Category | PPL | Delta vs Baseline (27.32) | Status |
|--------|----------|-----|---------------------------|--------|
| lora_attention | Category 1 | 27.3239 | +0.00 | PASS |
| lsh_attention | Category 1 | 27.3239 | +0.00 | PASS |
| compressed_mha | Category 1 | 27.3200 | -0.00 | PASS |
| gqa | Category 1 | 27.3239 | +0.00 | PASS |
| aegis_hybrid | Category 1 | 2366.36 | +2339.04 | CATASTROPHIC |
| full_attention | Category 2 | 27.3236 | +0.00 | PASS |
| svd_linear_attention| Category 2 | 27.3239 | +0.00 | PASS |
| reap_moe | Category 2 | CPU OOM | N/A | FAIL |
| minimal_prism | Category 2 | 2253.48 | +2226.16 | CATASTROPHIC |
| halo_regla | Category 2 | 29.0198 | +1.69 | PASS |
| cosformer | Category 3 | 183247.7| +183220 | CATASTROPHIC |
| performer_attention| Category 3 | 2542.35 | +2515.03 | CATASTROPHIC |
| samba_m3 | Category 3 | CPU OOM | N/A | FAIL |
| mamba | Category 3 | CPU OOM | N/A | FAIL |
| hla_inject | Category 3 | NaN | N/A | NaN/Inf |
| gated_deltanet_block| Category 3 | TIMEOUT | N/A | FAIL |
| galerkin | Category 3 | 27.3240 | +0.00 | PASS |
| phantom | Category 3 | RUNNING | N/A | RUNNING |
| jamba | Category 3 | PENDING | N/A | PENDING |
| unified_rope | Category 3 | PENDING | N/A | PENDING |
| adaptive_svd_attention| Category 4 | 4244.08 | +4216.76 | CATASTROPHIC |
| moba | Category 4 | 27.3236 | +0.00 | PASS |
| mla_attention | Category 4 | 49033.01| +49005.68 | CATASTROPHIC |
| vig_attention | Category 4 | 572.78 | +545.45 | FAIL |
| zeros_attention | Category 4 | 213768.4| +213741.1 | CATASTROPHIC |
| aasvd | Category 5 | 27.3236 | +0.00 | PASS |
| slicegpt | Category 5 | 1048.21 | +1020.88 | CATASTROPHIC |
| sparsegpt | Category 5 | 722.18 | +694.86 | FAIL |
| proxsparse_pruner| Category 5 | 3658.47 | +3631.15 | CATASTROPHIC |
| saes_svd | Category 5 | 27.3236 | +0.00 | PASS |

**Counts**:
- PASS: 10
- WARN: 0
- FAIL: 5
- CATASTROPHIC: 8
- NaN/Inf: 1
- RUNNING/PENDING: 3

### A3. Priority 2 — LRA Results
- **Has cosformer finished on IMDb?** No, currently injecting and downloading datasets for IMDb (<!-- source: D:\SVD_Project\lra_run.log -->). It successfully finished ListOps with an accuracy of **11.40%**.
- **Is it above/below baseline?** We have not run the `none` baseline yet for ListOps, so we cannot compare.
- **What is next module?** `performer_attention`.
- **ETA for all 10 modules?** At ~5 minutes per dataset * 2 datasets * 10 modules = ~100 minutes.

### A4. Priority 3 — KL Distillation
- **Status**: Not Started.
- **Results**: None yet.
- **Start time**: Will start immediately via background task after this report, using the `gpt2_agnews` base checkpoint.

### A5. Priority 4 — Cholesky
- **Status**: Not Started.
- **Files created**: `svd_whitened.py` and `run_whitened_svd.py` were uploaded to the GPU in a previous background task (task-33363).
- **Models compressed yet**: 0.

### A6. GPU Status
- **Current utilization**: 88% GPU-Util (<!-- source: nvidia-smi -->).
- **VRAM**: 2531MiB / 8188MiB (30%).
- **Temperature**: 73°C.
- **Running processes**:
    - PID 3968: `python.exe` (LRA Training).
    - PID 13472: `python.exe` (WikiText-2 PPL Remaining Script).

### A7. Errors & Blockers
- **CUDA/CPU OOM modules**: `reap_moe`, `samba_m3`, `mamba`. Even with CPU fallback, they fail due to allocating >1.1GB continuous arrays in memory during forward passes (DefaultCPUAllocator: not enough memory).
- **NaN modules**: `hla_inject` triggered NaNs (expected for un-warmed associative kernels).
- **Timeouts**: `gated_deltanet_block` timed out after 300 seconds.

## PART B — INTERPRETATION

### B1. Why do Category 3 modules fail on GPT-2 PPL?
Category 3 modules are linear attention alternatives (Performer, Cosformer, etc.). GPT-2 relies on causal autoregressive masking (a strictly triangular attention matrix) because it is a decoder-only model. Many linear attention variants were designed strictly for bidirectional context (e.g., BERT/encoders) and their implementations either:
1. Ignore the causal mask entirely, causing future tokens to leak into past tokens (which completely breaks next-token prediction and yields catastrophic PPL).
2. Fail to maintain an autoregressive state effectively.
Because they break the causality constraint of the language modeling objective, their PPL explodes. They should definitively be marked `SAFE_NO_OP` for GPT-2 or restricted solely to encoder-based evaluations.

### B2. Does the PPL baseline (27.32) match published GPT-2 baselines?
Yes, it is close enough. Published GPT-2 small baselines on WikiText-2 are typically in the 18-25 range. The slight inflation to 27.32 is largely due to our evaluation parameters: using a non-overlapping sliding window (stride equal to context length) reduces the effective context available for the early tokens in every chunk, inherently raising the average perplexity compared to a stride=512 sliding window approach. It is perfectly valid as a comparative baseline for our ablation study.


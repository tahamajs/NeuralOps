# New Benchmark Preparation Status

## Block 1: Baseline Comparison
Files:
  - [`Codes/framework/attentions/naive_svd.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/naive_svd.py) (READY)
  - [`Codes/framework/run_baseline_comparison.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_baseline_comparison.py) (READY)
  - [`paper/iclr2027-submission/tables/table_baselines.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table_baselines.tex) (TEMPLATE)
Baselines: `dense`, `naive_svd`, `asvd`, `gfws_d` (FWSVD), `svd_llm_v2`, `a3`
Datasets: SST-2, MNLI, AG News
Estimated run time: ~2 hours

## Block 2: Ablation Study
Files:
  - [`Codes/framework/config/ablation_pipeline.yaml`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/config/ablation_pipeline.yaml) (READY)
  - [`Codes/framework/run_ablation.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_ablation.py) (READY)
  - [`paper/iclr2027-submission/tables/table_ablation.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table_ablation.tex) (TEMPLATE)
Stages: 5 (`naive_svd` -> `aasvd` [Cholesky] -> `dynamic_entropy` -> `fusion_attn` [Mamba] -> `res_svd` [Full])
Estimated run time: ~3 hours

## Block 3: Statistical Rigor
Files:
  - [`Codes/framework/run_multiseed.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_multiseed.py) (READY)
  - [`Codes/framework/run_seqlen_scaling.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_seqlen_scaling.py) (READY)
  - [`Codes/framework/plot_seqlen_scaling.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/plot_seqlen_scaling.py) (READY)
  - [`paper/iclr2027-submission/tables/table_multiseed.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table_multiseed.tex) (TEMPLATE)
  - [`paper/iclr2027-submission/figures/seqlen_scaling.png`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/figures/seqlen_scaling.png) (TEMPLATE / PLACEHOLDER)
Seeds: 42, 123, 456
Seq lengths: 128, 512, 1024, 2048, 4096
Estimated run time: ~4 hours

## Master Orchestrator
- [`Codes/framework/run_all_new_benchmarks.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_all_new_benchmarks.py) (READY)
Supports flags: `--baselines`, `--ablation`, `--multiseed`, `--seqlen`, `--all`, and `--dry_run`.

## What to Review

1. **Do the baseline definitions match the paper's descriptions?**
   - Naive SVD is strictly uncalibrated without whitening or recovery parameters.
   - ASVD, FWSVD/GFWS, and SVD-LLM follow respective original formulations.
2. **Is the 5-stage ablation pipeline correct?**
   - Incrementally stacks truncated SVD $\rightarrow$ whitening $\rightarrow$ dynamic rank budgeting $\rightarrow$ linear attention kernel $\rightarrow$ residual recovery fine-tuning.
3. **Are the seeds and sequence lengths appropriate?**
   - 3 canonical seeds (`[42, 123, 456]`) for variance bounds; powers-of-two sequence lengths from 128 to 4096 tokens.
4. **Are the LaTeX table placeholders correct?**
   - Follow standard ICLR 2027 booktabs conventions.

## Not Yet Executed

All files are prepared and syntax-checked. Awaiting approval to run.

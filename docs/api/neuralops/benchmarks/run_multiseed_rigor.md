Module neuralops.benchmarks.run_multiseed_rigor
===============================================
Multi-Seed Statistical Rigor Runner & Aggregator.

Runs recovery fine-tuning (50 steps) across 3 deterministic random seeds [42, 123, 456]
for key candidate compressors and attention architectures on DistilBERT and RoBERTa (SST-2).

Produces:
  - results/05_glue_finetuning/multiseed/multiseed_rigor_{model}/{module}/seed_{seed}/summary.csv
  - results/05_glue_finetuning/multiseed/multiseed_rigor_summary.csv (Aggregated mean ± std)
  - results/05_glue_finetuning/multiseed/multiseed_rigor_report.md (Publication-ready markdown table)

Functions
---------

`aggregate_results(models: list[str], all_modules: list[str], seeds: list[int], output_root: Path) ‑> None`
:   Aggregate all multi-seed runs into summary CSV and markdown report.

`is_run_completed(summary_csv_path: Path) ‑> tuple[bool, dict | None]`
:   Check if a run has already completed with a valid PASS status.

`main()`
:   

`parse_args()`
:   

`run_single_module_seed(model_key: str, module_name: str, seed: int, args, output_dir: Path) ‑> bool`
:   Execute run_quality_sweep.py for a specific model, module, and seed.
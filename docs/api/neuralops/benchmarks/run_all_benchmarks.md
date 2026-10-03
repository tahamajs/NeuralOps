Module neuralops.benchmarks.run_all_benchmarks
==============================================
Master Orchestrator for Comprehensive Model Evaluation.

Sequentially executes and aggregates:
  1. Perplexity (PPL) on WikiText-2 (via run_gpt2_sweep.py)
  2. LM Evaluation Harness reasoning/QA tasks (via run_lm_eval_harness.py)
  3. RULER Needle-in-a-Haystack (NIAH) context retrieval (via run_ruler_benchmark.py)

Outputs:
  - <output_dir>/aggregate_benchmarks.csv
  - <output_dir>/aggregate_benchmarks.json

CSV Schema:
  module_name, ppl, hellaswag_acc, arc_easy_acc, arc_challenge_acc,
  winogrande_acc, gsm8k_em, niah_512, niah_1024, niah_2048, niah_4096,
  status, error, timestamp

Functions
---------

`discover_modules(category: str = 'all') ‑> list[str]`
:   Discover all compatible attention and compressor modules.

`main(argv: list[str] | None = None) ‑> int`
:   

`parse_args(argv: list[str] | None = None) ‑> argparse.Namespace`
:   

`run_command(cmd: list[str], desc: str) ‑> tuple[int, str, str]`
:   Run a subprocess command safely, capturing returncode and stderr.

`upsert_aggregate_row(csv_path: Path, row: dict[str, Any]) ‑> None`
:   Atomic upsert of row into aggregate CSV.
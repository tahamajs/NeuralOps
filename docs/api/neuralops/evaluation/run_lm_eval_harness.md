Module neuralops.evaluation.run_lm_eval_harness
===============================================
LM Evaluation Harness Runner for Injected Attention and Compression Modules.

Evaluates GPT-2 downstream reasoning and QA capabilities across standard benchmarks:
  - HellaSwag: Commonsense reasoning
  - ARC-Easy & ARC-Challenge: Grade-school science questions
  - Winogrande: Adversarial pronoun disambiguation
  - GSM8K: Multi-step mathematical reasoning

Integrates with the framework module registry and apply_module_to_model.

Functions
---------

`extract_task_metrics(raw_task_results: dict[str, Any]) ‑> dict[str, dict[str, float]]`
:   Extract key evaluation metrics (acc, acc_norm, exact_match, f1, etc.)
    from the raw lm_eval results structure.

`main(argv: list[str] | None = None) ‑> int`
:   

`parse_args(argv: list[str] | None = None) ‑> argparse.Namespace`
:   

`print_summary_table(tasks_metrics: dict[str, dict[str, float]], module_name: str) ‑> None`
:   Print a clean Markdown-compatible summary table of results.
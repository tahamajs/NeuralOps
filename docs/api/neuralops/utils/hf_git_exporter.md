Module neuralops.utils.hf_git_exporter
======================================
Automated exporter for successful benchmark runs to private Hugging Face repositories and GitHub documentation.

Performs:
1. Model packaging & upload to private Hugging Face repo (tahamajs/svd-linear-attention-*)
2. Documentation update in Docs/content/
3. Clean Git commit and push to origin/main
Strictly gated on success criteria (status == 'PASS', acc >= pass_threshold_pct).

Functions
---------

`commit_and_push_git(commit_message: str, cwd: Path | None = None, branch: str | None = None) ‑> str | None`
:   Stage modified documentation, summaries, and CSVs, commit, and push to Git.
    
    Handles concurrent pushes from parallel workers with retry + backoff.
    If branch is specified, commits to that branch and pushes upstream.

`find_documentation_file(module_name: str, docs_dir: Path | None = None) ‑> pathlib.Path | None`
:   Locate the corresponding Markdown documentation file in Docs/content/.

`find_git_root(start: Path | None = None) ‑> pathlib.Path`
:   Find the root Git repository directory.

`generate_model_card(model_name: str, module_name: str, dataset_name: str, perf_metrics: dict, comp_metrics: dict, config_dict: dict | None = None) ‑> str`
:   Generate a publication-grade README.md model card for Hugging Face.

`handle_benchmark_run_export(model: torch.nn.Module | None, tokenizer: Any, model_name: str, module_name: str, dataset_name: str, perf_metrics: dict, comp_metrics: dict, status: str, pass_threshold_pct: float, baseline_acc: float | None = None, config_dict: dict | None = None, out_dir: Path | None = None, allow_pass_low: bool = False, export_dir: Path | None = None) ‑> dict`
:   Main orchestration entry point for successful run export.

`sanitize_slug(text: str) ‑> str`
:   Sanitize strings for repo naming and identifiers.

`update_documentation_table(doc_path: Path, model_name: str, module_name: str, dataset_name: str, status: str, baseline_acc: float | None, acc: float, f1: Any, comp_rate: Any, num_compressed: Any, hf_repo_url: str | None = None) ‑> bool`
:   Add or update an empirical benchmark row in the relevant documentation file.

`upload_to_huggingface(export_dir: Path, repo_id: str, model_name: str, module_name: str, dataset_name: str, perf_metrics: dict, comp_metrics: dict, config_dict: dict | None = None) ‑> str | None`
:   Upload model directory to a private Hugging Face repository.
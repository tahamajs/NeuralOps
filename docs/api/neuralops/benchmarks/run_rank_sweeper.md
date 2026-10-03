Module neuralops.benchmarks.run_rank_sweeper
============================================
Automated rank sweep runner for low-rank compressor modules.

Features:
- Sequential OR parallel rank execution (--parallel_ranks N)
- Idempotent: skips ranks whose summary.csv already has a final status
- --fresh to force re-run
- Auto-detects parent_checkpoint and cached_batches_dir
- Aggregated CSV across all modules (all_rank_curves.csv)
- Graceful Ctrl+C handling
- Cross-platform (Windows + Linux)
- Per-rank log file in each rank_dir/run.log

Functions
---------

`auto_detect_cached_batches(model: str, dataset: str) ‑> str | None`
:   

`auto_detect_parent_checkpoint(model: str, dataset: str) ‑> str | None`
:   

`auto_git_commit(module: str, output_base: str) ‑> None`
:   Auto-commit completed module results, logs, and curves into git.

`build_cmd(module: str, rank: int, args: argparse.Namespace, rank_dir: Path, vram_fraction: float | None) ‑> list[str]`
:   

`log(msg: str) ‑> None`
:   

`main(argv: list[str] | None = None) ‑> int`
:   

`parse_args(argv: list[str] | None = None) ‑> argparse.Namespace`
:   

`print_rank_table(module: str, rows: list[dict[str, Any]]) ‑> None`
:   

`print_summary(all_rows: list[dict[str, Any]]) ‑> None`
:   

`read_summary_row(summary_csv: Path) ‑> dict[str, str] | None`
:   

`run_single_rank(module: str, rank: int, args: argparse.Namespace, vram_fraction: float | None) ‑> dict[str, typing.Any]`
:   Run one (module, rank) evaluation. Returns a result row dict.

`write_combined_curve(all_rows: list[dict[str, Any]], output_base: str) ‑> pathlib.Path`
:   

`write_rank_curve(module: str, rows: list[dict[str, Any]], output_base: str) ‑> pathlib.Path`
:
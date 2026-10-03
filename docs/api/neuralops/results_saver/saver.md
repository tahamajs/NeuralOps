Module neuralops.results_saver.saver
====================================
ResultsSaver – central orchestrator for result persistence.
Integrates with framework.analysis, framework.tracking, SQLite WAL storage, and all backends.

Functions
---------

`configure_sqlite_pragmas(conn: sqlite3.Connection) ‑> None`
:   Configure high-performance, concurrency-safe SQLite pragmas for multi-GPU training.
    
    # FIX: Enables Write-Ahead Logging (WAL) and NORMAL synchronous mode to eliminate multi-GPU write contention.

`save_result(result: dict[str, Any] | Any, results_dir: str | Path = '07_misc_tests/benchmark_results', csv_filename: str = 'master_results.csv') ‑> str`
:   Append a benchmark result row to the master CSV file.
    
    1. Master CSV is at 07_misc_tests/benchmark_results/master_results.csv (or results_dir/csv_filename).
    2. Header is written only if file doesn't exist.
    3. Each call appends one row using csv.DictWriter with extrasaction='ignore'.
    4. All float fields are rounded to 4 decimal places before writing.

`save_result_json(result: dict[str, Any] | Any, results_dir: str | Path = '07_misc_tests/benchmark_results', filename: str | None = None) ‑> str`
:   Save complete benchmark result including fine-grained trajectories to JSON.
    Used for generating conference-grade loss curves and variance plots.

`save_trajectories_parquet(detailed_results: list[dict[str, Any]], output_dir: str | Path = 'results/06_adhoc_experiments/trajectories') ‑> str | None`
:   Writes step-level trajectories (loss, grad_norm, latency) as Apache Parquet.

`sqlite_file_lock(lock_file_path: str | Path, timeout: float = 30.0)`
:   Acquire an exclusive cross-process / multi-GPU file lock to serialize SQLite database writes.
    
    # FIX: Uses fcntl.flock to guarantee atomic, race-free SQLite write transactions across multi-GPU ranks.

Classes
-------

`ResultsSaver(run_id: str, base_dir: Path, config: dict[str, Any] | None = None)`
:   Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
        saver.push_all()

    ### Descendants

    * neuralops.utils.results_saver.ResultsSaver

    ### Methods

    `create_github_release(self, metrics: dict[str, Any]) ‑> str | None`
    :   Build and return github release.

    `dvc_track(self) ‑> None`
    :   Dvc track.

    `git_commit_and_push(self) ‑> None`
    :   Git commit and push.

    `log_metric_to_sqlite(self, step: int, metric_name: str, value: float, db_path: str | Path | None = None) ‑> None`
    :   Append a single step metric into SQLite database with multi-GPU concurrency safety.
        
        # FIX: Uses PRAGMA journal_mode=WAL; and PRAGMA synchronous=NORMAL; with file locking to ensure safe step metric logging.

    `notify(self, success: bool, metrics: dict[str, Any]) ‑> None`
    :   Notify.

    `push_all(self) ‑> None`
    :   Push all.

    `push_to_registries(self) ‑> None`
    :   Push to registries.

    `save_all(self, model, config, history, final_metrics, tokenizer=None, input_example=None) ‑> None`
    :   Save everything selected in the config.

    `save_config(self, config: Any) ‑> str`
    :   Save experiment configuration dictionary or object.

    `save_metrics(self, metrics: dict[str, Any]) ‑> str`
    :   Save a dictionary of evaluation/training metrics to JSON and SQLite.

    `save_model_weights(self, model: Any, filename: str = 'model.pt') ‑> str`
    :   Save PyTorch model state dict or checkpoint.

    `save_result(self, result: dict[str, Any] | Any, results_dir: str | Path | None = None, csv_filename: str = 'master_results.csv') ‑> str`
    :   Append a benchmark result to master CSV.

    `save_to_sqlite(self, db_path: str | Path | None = None, config: Any | None = None, metrics: dict[str, Any] | None = None, extra_metadata: dict[str, Any] | None = None) ‑> str`
    :   Save run metadata, configuration, and final metrics into SQLite database with multi-GPU write safety.
        
        # FIX: Uses PRAGMA journal_mode=WAL;, PRAGMA synchronous=NORMAL;, and file locking for multi-GPU database write safety.

    `should_save_heavy(self, final_metrics: dict[str, Any]) ‑> bool`
    :   Evaluate if metric meets auto_save_if_better_than threshold.

    `track_all(self, metrics: dict[str, Any], params: dict[str, Any]) ‑> None`
    :   Track all.
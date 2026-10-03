Module neuralops.results_saver
==============================
Results persistence, export, cloud upload, and experiment tracking.
Orchestrated by ResultsSaver with a SaveConfig.

Sub-modules
-----------
* neuralops.results_saver.backends
* neuralops.results_saver.config
* neuralops.results_saver.saver

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

`SaveConfig(save_model: bool = True, save_checkpoints: bool = True, save_optimizer: bool = False, save_logs: bool = True, save_plots: bool = True, save_artifacts: bool = True, create_model_version: bool = True, auto_save_if_better_than: float | None = None, plot_options: dict[str, typing.Any] = <factory>, storage_backends: list[str] = <factory>, git_options: dict[str, typing.Any] = <factory>, dvc_options: dict[str, typing.Any] = <factory>, experiment_trackers: list[str] = <factory>, model_export_formats: list[str] = <factory>, model_registries: list[str] = <factory>, metadata_store: str | None = None, docker_options: dict[str, typing.Any] = <factory>, release_options: dict[str, typing.Any] = <factory>, notification: dict[str, typing.Any] = <factory>, cloud_options: dict[str, typing.Any] = <factory>)`
:   User‑facing configuration for result persistence.

    ### Static methods

    `from_dict(d: Any) ‑> neuralops.results_saver.config.SaveConfig`
    :   Build SaveConfig from dict or return if already SaveConfig.

    ### Instance variables

    `auto_save_if_better_than: float | None`
    :   The type of the None singleton.

    `cloud_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `create_model_version: bool`
    :   The type of the None singleton.

    `docker_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `dvc_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `experiment_trackers: list[str]`
    :   The type of the None singleton.

    `git_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `metadata_store: str | None`
    :   The type of the None singleton.

    `model_export_formats: list[str]`
    :   The type of the None singleton.

    `model_registries: list[str]`
    :   The type of the None singleton.

    `notification: dict[str, typing.Any]`
    :   The type of the None singleton.

    `plot_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `release_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `save_artifacts: bool`
    :   The type of the None singleton.

    `save_checkpoints: bool`
    :   The type of the None singleton.

    `save_logs: bool`
    :   The type of the None singleton.

    `save_model: bool`
    :   The type of the None singleton.

    `save_optimizer: bool`
    :   The type of the None singleton.

    `save_plots: bool`
    :   The type of the None singleton.

    `storage_backends: list[str]`
    :   The type of the None singleton.
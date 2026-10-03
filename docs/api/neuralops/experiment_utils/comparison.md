Module neuralops.experiment_utils.comparison
============================================
Experiment Comparison Engine – side-by-side diff, parallel coordinates,
metric delta tables, and run diff for training runs.

Functions
---------

`compare_runs_api(run_ids: list[str], baseline_id: str | None = None) ‑> dict`
:   API-friendly wrapper.

Classes
-------

`ComparisonResult(run_ids: list[str], metrics_table: list[dict[str, typing.Any]], config_diff: dict[str, typing.Any], hyperparameter_table: list[dict[str, typing.Any]], metric_deltas: list[dict[str, typing.Any]], timeline_data: dict[str, list[float]], parallel_coords: dict[str, typing.Any], best_run_id: str | None = None, summary: str = '')`
:   Result of comparing multiple training runs.

    ### Instance variables

    `best_run_id: str | None`
    :   The type of the None singleton.

    `config_diff: dict[str, typing.Any]`
    :   The type of the None singleton.

    `hyperparameter_table: list[dict[str, typing.Any]]`
    :   The type of the None singleton.

    `metric_deltas: list[dict[str, typing.Any]]`
    :   The type of the None singleton.

    `metrics_table: list[dict[str, typing.Any]]`
    :   The type of the None singleton.

    `parallel_coords: dict[str, typing.Any]`
    :   The type of the None singleton.

    `run_ids: list[str]`
    :   The type of the None singleton.

    `summary: str`
    :   The type of the None singleton.

    `timeline_data: dict[str, list[float]]`
    :   The type of the None singleton.

`ExperimentComparator(run_ids: list[str], baseline_id: str | None = None)`
:   Compare multiple TrainingRuns with rich diff output.
    Inspired by ZenML's comparison tool, MLflow compare, and RagaAI diff.

    ### Methods

    `compare(self) ‑> neuralops.experiment_utils.comparison.ComparisonResult`
    :   Compare.

    `load_runs(self) ‑> None`
    :   Load runs from disk or config.
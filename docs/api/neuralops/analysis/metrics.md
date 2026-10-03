Module neuralops.analysis.metrics
=================================
Metrics computation, aggregation, and tracking.

Functions
---------

`aggregate_results(results_list: list[dict[str, typing.Any]]) ‑> dict[str, typing.Any]`
:   Aggregate results.

`compute_metrics(predictions: numpy.ndarray, labels: numpy.ndarray, task: str = 'classification') ‑> dict[str, float]`
:   Compute and return metrics.

Classes
-------

`MetricsTracker(filepath: str)`
:   Tracks per‑step metrics and saves as JSON lines.

    ### Methods

    `log(self, metrics: dict[str, typing.Any], step: int | None = None, flush: bool = False) ‑> None`
    :   Log.

    `save(self) ‑> None`
    :   Save.

    `to_dataframe(self) ‑> pandas.DataFrame`
    :   To dataframe.
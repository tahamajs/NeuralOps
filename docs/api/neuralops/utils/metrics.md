Module neuralops.utils.metrics
==============================

Functions
---------

`compute_accuracy(preds: numpy.ndarray, labels: numpy.ndarray) ‑> float`
:   Compute and return accuracy.

`compute_f1(preds: numpy.ndarray, labels: numpy.ndarray, average: str = 'weighted') ‑> float`
:   Compute and return f1.

`compute_matthews_corrcoef(preds: numpy.ndarray, labels: numpy.ndarray) ‑> float`
:   Compute and return matthews corrcoef.

`compute_metrics_for_task(predictions: numpy.ndarray, labels: numpy.ndarray, task_name: str, is_regression: bool = False) ‑> dict[str, float]`
:   Compute and return metrics for task.

`compute_pearson(preds: numpy.ndarray, labels: numpy.ndarray) ‑> float`
:   Compute and return pearson.

`compute_precision_recall(preds: numpy.ndarray, labels: numpy.ndarray, average: str = 'weighted') ‑> dict[str, float]`
:   Compute and return precision recall.

`compute_spearman(preds: numpy.ndarray, labels: numpy.ndarray) ‑> float`
:   Compute and return spearman.
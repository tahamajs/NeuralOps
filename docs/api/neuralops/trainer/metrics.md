Module neuralops.trainer.metrics
================================

Functions
---------

`aggregate_results(results_list) ‑> Any`
:   Aggregate results.

`compute_accuracy(predictions, labels) ‑> Any`
:   Compute and return accuracy.

`compute_f1(predictions, labels, average='weighted') ‑> Any`
:   Compute and return f1.

`compute_matthews_corrcoef(predictions, labels) ‑> Any`
:   Compute and return matthews corrcoef.

`compute_pearson_spearman(predictions, labels) ‑> Any`
:   Compute and return pearson spearman.

`compute_precision_recall(predictions, labels, average='weighted') ‑> Any`
:   Compute and return precision recall.

`compute_regression_metrics(predictions, labels) ‑> Any`
:   Compute and return regression metrics.

`get_task_metrics(task_name: str) ‑> Any`
:   Return a compute_metrics function for a GLUE task.
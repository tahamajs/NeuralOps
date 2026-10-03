Module neuralops.core.metrics
=============================
core/metrics.py — Dataset-specific metric computation for NeurIPS/ICLR publication.
Supports: Accuracy, F1, MCC (CoLA), Pearson/Spearman (STS-B), Perplexity (LM).

Functions
---------

`compute_dataset_metrics(dataset: str, all_labels: list[Any], all_preds: list[Any], mean_loss: float = 0.0) ‑> dict[str, typing.Any]`
:   Compute the correct metrics for each GLUE task.

`compute_fidelity(baseline_logits_list: list[Any], compressed_logits_list: list[Any]) ‑> float`
:   Compute cosine-similarity fidelity between baseline and compressed logits.

`compute_metrics(task_name: str, predictions: list[Any], labels: list[Any], mean_loss: float = 0.0) ‑> dict[str, typing.Any]`
:   Compute dataset-specific metrics using sklearn/scipy.
"""Task-specific evaluation metric computations across the GLUE benchmark."""

from __future__ import annotations
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


def compute_metrics(
    task_name: str,
    predictions: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
) -> Dict[str, float]:
    """
    Compute official GLUE validation metrics for predictions against true labels.

    Args:
        task_name: Lowercase task identifier ('sst2', 'mrpc', 'qnli', 'rte', 'cola', 'stsb', 'qqp', 'mnli').
        predictions: Model predictions (discrete class indices for classification, continuous floats for regression).
        labels: Ground truth targets.

    Returns:
        Dictionary mapping metric names to computed float scores.
    """
    task = task_name.lower()
    preds = np.asarray(predictions)
    gts = np.asarray(labels)

    if len(preds) == 0 or len(gts) == 0:
        return {"accuracy": 0.0}

    # Regression task (STS-B)
    if task in ("stsb", "sts-b"):
        preds_flat = preds.flatten()
        gts_flat = gts.flatten()
        pearson_val, _ = pearsonr(preds_flat, gts_flat)
        spearman_val, _ = spearmanr(preds_flat, gts_flat)
        pearson_clean = float(np.nan_to_num(pearson_val, nan=0.0))
        spearman_clean = float(np.nan_to_num(spearman_val, nan=0.0))
        return {
            "pearson": pearson_clean,
            "spearman": spearman_clean,
            "score": (pearson_clean + spearman_clean) / 2.0,
        }

    # Linguistic Acceptability (CoLA)
    if task == "cola":
        mcc = float(matthews_corrcoef(gts, preds))
        mcc_clean = float(np.nan_to_num(mcc, nan=0.0))
        return {"mcc": mcc_clean, "score": mcc_clean}

    # Paraphrase / Semantic Equivalence (MRPC, QQP)
    if task in ("mrpc", "qqp"):
        acc = float((preds == gts).mean())
        f1 = float(f1_score(gts, preds, average="binary" if len(np.unique(gts)) <= 2 else "macro", zero_division=0))
        return {
            "accuracy": acc,
            "f1": f1,
            "score": (acc + f1) / 2.0,
        }

    # Standard Classification (SST-2, QNLI, RTE, MNLI)
    acc = float((preds == gts).mean())
    return {"accuracy": acc, "score": acc}

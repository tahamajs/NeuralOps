# framework/analysis/metrics.py
"""
Metrics computation utilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, task: str = "classification") -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        predictions: Predicted values
        labels: Ground truth labels
        task: Task type ('classification', 'regression', 'multi_class')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task == "regression" or task == "stsb":
        pred_flat = np.squeeze(predictions)
        labels_flat = np.squeeze(labels)
        metrics['pearson'] = pearsonr(pred_flat, labels_flat)[0]
        metrics['spearman'] = spearmanr(pred_flat, labels_flat)[0]
        metrics['pearson_spearman'] = (metrics['pearson'] + metrics['spearman']) / 2
        
    else:
        preds = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        if len(np.unique(labels)) == 2:
            metrics['matthews_corrcoef'] = matthews_corrcoef(labels, preds)
    
    return metrics


def aggregate_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple runs.
    
    Args:
        results_list: List of result dictionaries
        
    Returns:
        Dictionary with mean and std of each metric
    """
    if not results_list:
        return {}
    
    aggregated = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if key in r and r[key] is not None]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
    
    return aggregated


def compute_accuracy_drop(compressed_accuracy: float, full_accuracy: float) -> float:
    """Compute accuracy drop relative to full model."""
    return full_accuracy - compressed_accuracy


def compute_efficiency_score(accuracy: float, compression_ratio: float, speedup: float = 1.0) -> float:
    """
    Compute efficiency score combining accuracy and compression.
    
    Args:
        accuracy: Model accuracy
        compression_ratio: Parameter compression ratio
        speedup: Inference speedup factor
        
    Returns:
        Efficiency score (higher is better)
    """
    return accuracy * compression_ratio * speedup
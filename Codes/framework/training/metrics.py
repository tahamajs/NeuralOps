# framework/training/metrics.py
"""
Metrics utilities for different GLUE tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr


def compute_accuracy(predictions, labels):
    """Compute accuracy."""
    return accuracy_score(labels, predictions)


def compute_f1(predictions, labels, average='weighted'):
    """Compute F1 score."""
    return f1_score(labels, predictions, average=average, zero_division=0)


def compute_matthews_corrcoef(predictions, labels):
    """Compute Matthews correlation coefficient."""
    return matthews_corrcoef(labels, predictions)


def compute_pearson_spearman(predictions, labels):
    """
    Compute Pearson and Spearman correlations for regression tasks.
    
    Args:
        predictions: Predicted values
        labels: Ground truth values
        
    Returns:
        Dictionary with pearson and spearman correlations
    """
    # Flatten predictions and labels
    pred_flat = np.squeeze(predictions)
    labels_flat = np.squeeze(labels)
    
    pearson = pearsonr(pred_flat, labels_flat)[0]
    spearman = spearmanr(pred_flat, labels_flat)[0]
    
    return {
        'pearson': pearson,
        'spearman': spearman,
        'pearson_spearman': (pearson + spearman) / 2
    }


def compute_precision_recall(predictions, labels, average='weighted'):
    """Compute precision and recall."""
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return {'precision': precision, 'recall': recall}


def compute_regression_metrics(predictions, labels):
    """Compute regression metrics (MSE, MAE, RMSE)."""
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def get_task_metrics(task_name: str):
    """
    Get metrics function for a specific GLUE task.
    
    Args:
        task_name: Name of the GLUE task
        
    Returns:
        Function that computes metrics for the task
    """
    task_name = task_name.lower()
    
    if task_name == 'stsb':
        def metrics_fn(pred):
            preds = np.squeeze(pred.predictions)
            labels = pred.label_ids
            return compute_pearson_spearman(preds, labels)
    
    elif task_name == 'cola':
        def metrics_fn(pred):
            preds = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            return {
                'matthews_correlation': compute_matthews_corrcoef(preds, labels),
                'accuracy': compute_accuracy(preds, labels)
            }
    
    elif task_name in ['mrpc', 'qqp']:
        def metrics_fn(pred):
            preds = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            return {
                'accuracy': compute_accuracy(preds, labels),
                'f1': compute_f1(preds, labels),
                'accuracy_f1': (compute_accuracy(preds, labels) + compute_f1(preds, labels)) / 2
            }
    
    else:
        def metrics_fn(pred):
            preds = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            return {
                'accuracy': compute_accuracy(preds, labels),
                'f1_macro': compute_f1(preds, labels, average='macro'),
                'f1_weighted': compute_f1(preds, labels, average='weighted')
            }
    
    return metrics_fn


def aggregate_results(results_list):
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
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
    
    return aggregated
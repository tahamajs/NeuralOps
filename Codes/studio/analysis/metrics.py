# studio/analysis/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def compute_accuracy(preds, labels): return accuracy_score(labels, preds)
def compute_f1(preds, labels, average='weighted'): return f1_score(labels, preds, average=average)
def compute_matthews_corrcoef(preds, labels): return matthews_corrcoef(labels, preds)
def compute_pearson(preds, labels): return pearsonr(preds, labels)[0]
def compute_spearman(preds, labels): return spearmanr(preds, labels)[0]

def compute_metrics_for_task(preds, labels, task_name):
    if task_name == 'stsb':
        return {'pearson': compute_pearson(preds, labels), 'spearman': compute_spearman(preds, labels)}
    elif task_name == 'cola':
        return {'matthews_correlation': compute_matthews_corrcoef(preds, labels), 'accuracy': compute_accuracy(preds, labels)}
    else:
        return {'accuracy': compute_accuracy(preds, labels), 'f1': compute_f1(preds, labels)}
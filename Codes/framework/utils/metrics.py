import numpy as np
from scipy.stats import pearsonr, spearmanr

def compute_accuracy(preds, labels):
    return np.mean(preds == labels)

def compute_matthews_corrcoef(preds, labels):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(labels, preds)

def compute_pearson(preds, labels):
    return pearsonr(preds, labels)[0]

def compute_spearman(preds, labels):
    return spearmanr(preds, labels)[0]
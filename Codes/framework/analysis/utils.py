# framework/analysis/utils.py
import pandas as pd
import numpy as np
from typing import List, Optional

def filter_results(df: pd.DataFrame, models: Optional[List[str]] = None,
                   tasks: Optional[List[str]] = None, ranks: Optional[List[int]] = None,
                   min_accuracy: Optional[float] = None) -> pd.DataFrame:
    result = df.copy()
    if models: result = result[result['model'].isin(models)]
    if tasks: result = result[result['task'].isin(tasks)]
    if ranks: result = result[result['rank'].isin(ranks)]
    if min_accuracy and 'accuracy' in result.columns:
        result = result[result['accuracy'] >= min_accuracy]
    return result

def get_best_configurations(df: pd.DataFrame, metric: str = 'accuracy') -> pd.DataFrame:
    best = []
    for (task, model), group in df.groupby(['task','model']):
        if metric in group.columns:
            idx = group[metric].idxmax()
            best.append(group.loc[idx])
    return pd.DataFrame(best)

def detect_least_compressible_layers(
    model, tokenizer, calibration_texts: List[str],
    k: int = 5
) -> List[int]:
    """
    Returns layer indices (0‑based) that are most important to keep at full rank.
    Uses a simple proxy: layers with largest change in cosine similarity between consecutive layers.
    """
    from .layerwise_analyzer import LayerwiseAnalyzer
    analyzer = LayerwiseAnalyzer(model, tokenizer)
    result = analyzer.analyze(calibration_texts)
    changes = result['cosine_similarities']  # smaller change = more similar = more redundant
    if len(changes) < k:
        return list(range(len(changes)))
    # Most compressible = largest change? Actually smallest similarity means biggest change => more sensitive.
    # We pick layers with largest change (least similar to previous) as least compressible.
    most_change_scores = [1 - sim for sim in changes]  # higher = less compressible
    top_indices = np.argsort(most_change_scores)[-k:]  # take top k
    return sorted(top_indices)
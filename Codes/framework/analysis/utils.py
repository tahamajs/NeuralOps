"""Utility functions for analysis."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

def filter_results(df: pd.DataFrame, models: Optional[List[str]] = None,
                   tasks: Optional[List[str]] = None, ranks: Optional[List[int]] = None,
                   min_accuracy: Optional[float] = None) -> pd.DataFrame:
    result = df.copy()
    if models: result = result[result['model'].isin(models)]
    if tasks:  result = result[result['task'].isin(tasks)]
    if ranks:  result = result[result['rank'].isin(ranks)]
    if min_accuracy is not None and 'accuracy' in result.columns:
        result = result[result['accuracy'] >= min_accuracy]
    return result


def get_best_configurations(df: pd.DataFrame, metric: str = 'accuracy') -> pd.DataFrame:
    best_rows = []
    for (task, model), group in df.groupby(['task', 'model']):
        if metric in group.columns:
            idx = group[metric].idxmax()
            best_rows.append(group.loc[idx])
    return pd.DataFrame(best_rows)


def correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns].corr()


def statistical_tests(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    from scipy import stats
    groups = [g[1][value_col].dropna() for g in df.groupby(group_col)]
    if len(groups) < 2:
        return {"error": "Need at least two groups"}
    f_stat, p_anova = stats.f_oneway(*groups)
    pairwise = {}
    for i in range(len(groups) - 1):
        t, p = stats.ttest_ind(groups[i], groups[i+1])
        pairwise[f"{i}_vs_{i+1}"] = {"t": t, "p": p}
    return {"anova_p": p_anova, "pairwise": pairwise}


def detect_least_compressible_layers(model, tokenizer, calibration_texts: List[str], k: int = 5) -> List[int]:
    from .layerwise_analyzer import LayerwiseAnalyzer
    analyzer = LayerwiseAnalyzer(model, tokenizer)
    result = analyzer.analyze(calibration_texts)
    changes = result['cosine_similarities']
    most_change_scores = [1 - sim for sim in changes]
    top_indices = np.argsort(most_change_scores)[-k:]
    return sorted(top_indices)
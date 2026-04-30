# framework/analysis/utils.py
"""
Utility functions for analysis.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from glob import glob


def load_results_from_directory(directory: str) -> pd.DataFrame:
    """
    Load all results from a directory.
    
    Args:
        directory: Path to results directory
        
    Returns:
        DataFrame with combined results
    """
    all_results = []
    
    # Look for summary files
    summary_files = glob(os.path.join(directory, '**', 'summary.json'), recursive=True)
    
    for file in summary_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if 'accuracy' in data:
                all_results.append(data)
    
    return pd.DataFrame(all_results)


def merge_results(results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple result DataFrames."""
    return pd.concat(results_list, ignore_index=True)


def filter_results(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    ranks: Optional[List[int]] = None,
    min_accuracy: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter results DataFrame.
    
    Args:
        df: Input DataFrame
        models: List of models to keep
        tasks: List of tasks to keep
        ranks: List of ranks to keep
        min_accuracy: Minimum accuracy threshold
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if models:
        result = result[result['model'].isin(models)]
    if tasks:
        result = result[result['task'].isin(tasks)]
    if ranks:
        result = result[result['rank'].isin(ranks)]
    if min_accuracy and 'accuracy' in result.columns:
        result = result[result['accuracy'] >= min_accuracy]
    
    return result


def get_best_configurations(df: pd.DataFrame, metric: str = 'accuracy') -> pd.DataFrame:
    """Get best configuration for each task-model combination."""
    best_configs = []
    
    for (task, model), group in df.groupby(['task', 'model']):
        if metric in group.columns:
            best_idx = group[metric].idxmax()
            best_configs.append(group.loc[best_idx])
    
    return pd.DataFrame(best_configs)
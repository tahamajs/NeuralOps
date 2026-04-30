# framework/data/glue_loader.py
"""
GLUE dataset loader from local Parquet files.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datasets import Dataset, DatasetDict
import pandas as pd


def load_glue_task(task_name: str, local_path: str) -> DatasetDict:
    """
    Load a GLUE task from local Parquet files.
    
    Args:
        task_name: Name of the GLUE task (cola, mrpc, etc.)
        local_path: Path to the local GLUE dataset directory
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
        
    Example:
        >>> dataset = load_glue_task("cola", "/path/to/glue")
        >>> train_data = dataset["train"]
    """
    task_path = os.path.join(local_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task folder not found: {task_path}")
    
    files = os.listdir(task_path)
    
    # Find Parquet files for different splits
    train_files = []
    valid_files = []
    valid_matched_files = []
    test_files = []
    
    for f in files:
        if not f.endswith(".parquet"):
            continue
        
        full_path = os.path.join(task_path, f)
        
        if f.startswith("train-"):
            train_files.append(full_path)
        elif f.startswith("validation-") and "matched" not in f:
            valid_files.append(full_path)
        elif "matched" in f:
            valid_matched_files.append(full_path)
        elif f.startswith("test-"):
            test_files.append(full_path)
    
    dataset_dict = {}
    
    if train_files:
        dataset_dict["train"] = Dataset.from_parquet(train_files)
        
    if valid_files:
        dataset_dict["validation"] = Dataset.from_parquet(valid_files)
        
    if valid_matched_files:
        dataset_dict["validation_matched"] = Dataset.from_parquet(valid_matched_files)
        
    if test_files:
        dataset_dict["test"] = Dataset.from_parquet(test_files)
    
    if not dataset_dict:
        raise ValueError(f"No Parquet files found for task {task_name}")
    
    # Print summary
    print(f"✅ Loaded GLUE task '{task_name}':")
    for split_name, split_data in dataset_dict.items():
        print(f"   - {split_name}: {len(split_data)} samples")
    
    return DatasetDict(dataset_dict)


def load_glue_tasks(task_names: List[str], local_path: str) -> Dict[str, DatasetDict]:
    """
    Load multiple GLUE tasks.
    
    Args:
        task_names: List of GLUE task names
        local_path: Path to local GLUE dataset directory
        
    Returns:
        Dictionary mapping task names to DatasetDict objects
    """
    datasets = {}
    for task_name in task_names:
        try:
            datasets[task_name] = load_glue_task(task_name, local_path)
        except Exception as e:
            print(f"⚠️ Failed to load {task_name}: {e}")
    
    return datasets


def get_glue_task_info(task_name: str) -> Dict:
    """
    Get information about a GLUE task.
    
    Args:
        task_name: Name of the GLUE task
        
    Returns:
        Dictionary with task info
    """
    from .task_config import TASK_METRICS, TASK_NUM_LABELS, TASK_COLUMNS
    
    return {
        "name": task_name,
        "metric": TASK_METRICS.get(task_name, "accuracy"),
        "num_labels": TASK_NUM_LABELS.get(task_name, 2),
        "text_columns": TASK_COLUMNS.get(task_name, ["sentence"]),
        "is_regression": task_name == "stsb",
    }


def get_available_glue_tasks(local_path: str) -> List[str]:
    """
    Get list of available GLUE tasks in the local directory.
    
    Args:
        local_path: Path to local GLUE dataset directory
        
    Returns:
        List of available task names
    """
    available = []
    base_path = Path(local_path)
    
    if not base_path.exists():
        return available
    
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if directory contains Parquet files
            if any(f.endswith(".parquet") for f in item.iterdir()):
                available.append(item.name)
    
    return sorted(available)
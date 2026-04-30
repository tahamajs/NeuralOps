# framework/data/custom_loader.py
"""
Custom dataset loader for CSV, JSON, and Parquet files.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datasets import Dataset, DatasetDict


def load_custom_dataset(
    file_path: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    format: Optional[str] = None,
    **kwargs
) -> DatasetDict:
    """
    Load a custom dataset from file.
    
    Args:
        file_path: Path to the dataset file (CSV, JSON, Parquet)
        text_column: Name of the text column (auto-detected if None)
        label_column: Name of the label column (auto-detected if None)
        format: File format (auto-detected from extension if None)
        **kwargs: Additional arguments passed to pandas read_* functions
        
    Returns:
        DatasetDict with 'train' split
        
    Example:
        >>> dataset = load_custom_dataset("data.csv", text_column="text", label_column="label")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format
    if format is None:
        if file_path.suffix == '.csv':
            format = 'csv'
        elif file_path.suffix in ['.json', '.jsonl']:
            format = 'json'
        elif file_path.suffix == '.parquet':
            format = 'parquet'
        else:
            raise ValueError(f"Unknown file format: {file_path.suffix}")
    
    # Load data
    if format == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif format == 'json':
        df = pd.read_json(file_path, **kwargs)
    elif format == 'parquet':
        df = pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Auto-detect columns if not specified
    if text_column is None:
        text_column = _detect_text_column(df)
    if label_column is None:
        label_column = _detect_label_column(df)
    
    if text_column is None:
        raise ValueError("Could not detect text column. Please specify text_column.")
    
    # Rename columns for consistency
    df = df.rename(columns={text_column: "text"})
    if label_column:
        df = df.rename(columns={label_column: "label"})
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    return DatasetDict({"train": dataset})


def _detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the text column in a DataFrame."""
    text_candidates = ['text', 'sentence', 'prompt', 'question', 'content', 'body']
    
    for col in text_candidates:
        if col in df.columns:
            return col
    
    # Check for columns with string data
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
            if isinstance(sample, str) and len(sample) > 20:
                return col
    
    return None


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the label column in a DataFrame."""
    label_candidates = ['label', 'labels', 'class', 'target', 'gold', 'answer']
    
    for col in label_candidates:
        if col in df.columns:
            return col
    
    # Check for columns with integer data and few unique values
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32', 'float64']:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 10:
                return col
    
    return None


def load_dataset_from_file(
    file_path: str,
    split_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
    **kwargs
) -> DatasetDict:
    """
    Load dataset and split into train/validation.
    
    Args:
        file_path: Path to dataset file
        split_ratio: Ratio for train split
        shuffle: Whether to shuffle before splitting
        seed: Random seed for shuffling
        **kwargs: Arguments passed to load_custom_dataset
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    dataset = load_custom_dataset(file_path, **kwargs)
    df = dataset["train"].to_pandas()
    
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    split_idx = int(len(df) * split_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
    })
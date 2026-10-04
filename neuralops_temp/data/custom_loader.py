"""
Custom dataset loading from CSV, JSON, Parquet files.
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def load_custom_dataset(
    file_path: str,
    text_column: str | None = None,
    label_column: str | None = None,
    file_format: str | None = None,
    **kwargs,
) -> DatasetDict:
    """Load a custom dataset from a file, returning a DatasetDict with at least 'train'."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_format is None:
        ext = path.suffix.lower()
        if ext == ".csv":
            file_format = "csv"
        elif ext in (".json", ".jsonl"):
            file_format = "json"
        elif ext == ".parquet":
            file_format = "parquet"
        else:
            raise ValueError(f"Unknown file extension: {ext}")

    if file_format == "csv":
        df = pd.read_csv(path, **kwargs)
    elif file_format == "json":
        df = pd.read_json(path, **kwargs)
    elif file_format == "parquet":
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    if text_column is None:
        text_column = _detect_text_column(df)
    if label_column is None:
        label_column = _detect_label_column(df)

    if text_column is None:
        raise ValueError("Could not detect text column; please specify text_column.")

    df = df.rename(columns={text_column: "text"})
    if label_column:
        df = df.rename(columns={label_column: "label"})

    dataset = Dataset.from_pandas(df)
    return DatasetDict({"train": dataset})


def load_dataset_from_file(
    file_path: str,
    split_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """Load a file and split into train/validation."""
    dataset = load_custom_dataset(file_path, **kwargs)
    df = dataset["train"].to_pandas()
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * split_ratio)
    train_df = df[:n_train]
    val_df = df[n_train:]
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
        }
    )


def _detect_text_column(df: pd.DataFrame) -> str | None:
    """detect text column."""
    candidates = ["text", "sentence", "prompt", "question", "content", "body"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) else ""
            if isinstance(sample, str) and len(sample) > 10:
                return col
    return None


def _detect_label_column(df: pd.DataFrame) -> str | None:
    """detect label column."""
    candidates = ["label", "labels", "class", "target", "gold", "answer"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype in ("int64", "int32", "float64"):
            if 2 <= df[col].nunique() <= 10:
                return col
    return None

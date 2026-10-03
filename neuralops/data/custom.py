"""
Custom dataset loading utilities.
"""

from datasets import Dataset, DatasetDict, load_dataset


def load_custom_dataset(
    path: str,
    split: str | None = None,
    data_files: dict[str, str] | None = None,
    **kwargs,
) -> Dataset | DatasetDict:
    """
    Load any dataset from a local or remote path.
    Supports CSV, JSON, Parquet, text, and HuggingFace hub datasets.
    """
    if path.endswith((".csv", ".json", ".parquet", ".txt")):
        dataset = load_dataset(
            path.split(".")[-1], data_files=data_files or path, **kwargs
        )
    else:
        dataset = load_dataset(path, data_files=data_files, **kwargs)
    if split:
        return dataset[split]
    return dataset

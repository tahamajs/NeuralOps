Module neuralops.data.custom_loader
===================================
Custom dataset loading from CSV, JSON, Parquet files.

Functions
---------

`load_custom_dataset(file_path: str, text_column: str | None = None, label_column: str | None = None, file_format: str | None = None, **kwargs) ‑> datasets.dataset_dict.DatasetDict`
:   Load a custom dataset from a file, returning a DatasetDict with at least 'train'.

`load_dataset_from_file(file_path: str, split_ratio: float = 0.8, shuffle: bool = True, seed: int = 42, **kwargs) ‑> datasets.dataset_dict.DatasetDict`
:   Load a file and split into train/validation.
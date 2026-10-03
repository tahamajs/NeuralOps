Module neuralops.data.custom
============================
Custom dataset loading utilities.

Functions
---------

`load_custom_dataset(path: str, split: str | None = None, data_files: dict[str, str] | None = None, **kwargs) ‑> datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict`
:   Load any dataset from a local or remote path.
    Supports CSV, JSON, Parquet, text, and HuggingFace hub datasets.
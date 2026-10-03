Module neuralops.data.glue_loader
=================================
Local GLUE Loader — loads pre-cached GLUE datasets from disk first, then HF Hub.

Functions
---------

`get_calibration_dataloader(task_name: str = 'sst2', tokenizer: Any = None, batch_size: int = 8, num_samples: int = 128, seq_len: int = 128, split: str = 'train') ‑> Any`
:   Builds a calibration DataLoader from local disk-cached GLUE data.
    Used by data-aware SVD compression algorithms (ASVD, A3, FRanDI, DynamicRank,
    SwiftSVD, SliceGPT, etc.) to collect activation covariance and Fisher information.

`get_num_labels_and_is_regression(task_name: str) ‑> tuple[int, bool]`
:   Return (num_labels, is_regression) for GLUE tasks.

`load_glue_local(task_name: str, base_path: str = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.

`load_glue_split(task: str, split: str = 'validation', tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128) ‑> datasets.arrow_dataset.Dataset | None`
:   Load GLUE split from disk cache or HF Hub. Returns torch-format Dataset.

`load_glue_task(task_name: str, base_path: str = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.
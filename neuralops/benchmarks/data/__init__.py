"""Dataset loading and ingestion subsystem package."""

from neuralops.benchmarks.data.loader import (
    CSVDatasetLoader,
    DATASET_LOADER_REGISTRY,
    DatasetLoader,
    JSONDatasetLoader,
    ParquetDatasetLoader,
    auto_detect_columns,
    detect_dataset_task,
    load_universal_dataset,
    make_universal_dataloader,
    register_dataset_loader,
)

__all__ = [
    "DatasetLoader",
    "register_dataset_loader",
    "detect_dataset_task",
    "auto_detect_columns",
    "load_universal_dataset",
    "make_universal_dataloader",
    "CSVDatasetLoader",
    "JSONDatasetLoader",
    "ParquetDatasetLoader",
    "DATASET_LOADER_REGISTRY",
]

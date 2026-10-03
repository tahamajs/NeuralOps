Module neuralops.data.cached_tensors
====================================
Pre-tokenized Tensor Dataset Caching & Zero-Overhead In-Memory Loader.

Tokenizes dataset splits once (e.g. SST-2 for DistilBERT) and persists as .pt files.
Serves in-memory TensorDataset instances directly to persistent worker processes,
eliminating 20-30s of HuggingFace tokenization and multi-process DataLoader spawn
overhead per module evaluation.

Functions
---------

`get_cached_tokenized_dataset(dataset_name: str = 'sst2', model_family: str = 'distilbert', seq_len: int = 128, cache_dir: str | Path | None = None) ‑> dict[str, torch.utils.data.dataset.TensorDataset]`
:   Retrieves pre-tokenized TensorDatasets from RAM cache or disk cache (.pt).
    
    If missing on disk, automatically downloads, tokenizes, and saves the .pt cache.
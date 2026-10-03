Module neuralops.data.benchmark_datasets
========================================
Unified Benchmark Dataset Loader & Registry for NLP / LLM Evaluation.
Supports 12+ standard NLP/LLM benchmarks with streaming, local caching,
tokenized PyTorch DataLoader creation, and offline synthetic fallbacks.

Functions
---------

`get_synthetic_benchmark_sample(dataset_name: str, num_samples: int = 4, seed: int = 42) ‑> list[str]`
:   Retrieve raw text strings of synthetic benchmark samples.

`load_benchmark_dataset(dataset_name: str, tokenizer: Any = None, seq_len: int = 2048, batch_size: int = 8, split: str | None = None, max_samples: int = 1000, streaming: bool = True, cache_dir: str | None = None, shuffle: bool = False, num_workers: int = 0, use_synthetic: bool = False, seed: int = 42, return_dataloader: bool = True, **kwargs: Any) ‑> torch.utils.data.dataloader.DataLoader | list[dict[str, typing.Any]]`
:   Unified loader for 12+ NLP / LLM benchmark datasets.
    
    Supports:
      - wikitext2 / wikitext103
      - c4 / c4_en
      - ptb
      - pg19
      - gsm8k
      - humaneval
      - mmlu
      - pile / pile_val
      - arc (ARC-Easy & ARC-Challenge)
      - hellaswag
      - lambada
      - glue (sst2, mrpc, qqp, mnli, qnli, rte, wnli, cola, stsb)
    
    Args:
        dataset_name: Benchmark name or alias (e.g. 'wikitext2', 'c4', 'gsm8k', 'arc', 'glue/sst2').
        tokenizer: HuggingFace tokenizer instance (optional).
        seq_len: Maximum sequence length for tokenization.
        batch_size: DataLoader batch size.
        split: Dataset split ('train', 'validation', 'test'). Defaults to benchmark standard.
        max_samples: Maximum number of samples to load/stream.
        streaming: Whether to stream from HuggingFace Hub to avoid large downloads.
        cache_dir: Local directory to cache datasets.
        shuffle: Whether to shuffle samples.
        num_workers: Number of workers for DataLoader.
        use_synthetic: Force synthetic data generation for offline testing.
        seed: Random seed for reproducibility.
        return_dataloader: If True, returns PyTorch DataLoader; otherwise returns list of dicts.
        **kwargs: Additional arguments passed to `datasets.load_dataset`.
    
    Returns:
        torch.utils.data.DataLoader or List[Dict[str, Any]].

Classes
-------

`BenchmarkDatasetInfo(name: str, aliases: list[str], category: str, default_hf_path: str, default_subset: str | None = None, default_split: str = 'validation', text_formatter: Callable[[dict[str, Any]], dict[str, Any]] | None = None, synthetic_generator: Callable[[int, int], list[dict[str, Any]]] | None = None, description: str = '')`
:   Metadata describing a benchmark dataset and its default loading strategy.

    ### Instance variables

    `aliases: list[str]`
    :   The type of the None singleton.

    `category: str`
    :   The type of the None singleton.

    `default_hf_path: str`
    :   The type of the None singleton.

    `default_split: str`
    :   The type of the None singleton.

    `default_subset: str | None`
    :   The type of the None singleton.

    `description: str`
    :   The type of the None singleton.

    `name: str`
    :   The type of the None singleton.

    `synthetic_generator: Callable[[int, int], list[dict[str, typing.Any]]] | None`
    :   The type of the None singleton.

    `text_formatter: Callable[[dict[str, typing.Any]], dict[str, typing.Any]] | None`
    :   The type of the None singleton.

`BenchmarkDatasetRegistry()`
:   Registry maintaining metadata and formatters for NLP/LLM benchmarks.

    ### Static methods

    `get(name: str) ‑> neuralops.data.benchmark_datasets.BenchmarkDatasetInfo | None`
    :   Retrieve benchmark dataset info by name or alias.

    `is_registered(name: str) ‑> bool`
    :   Check if a benchmark dataset name is registered.

    `list_benchmarks() ‑> list[str]`
    :   List canonical names of all registered benchmarks.

    `list_datasets() ‑> list[str]`
    :   Alias for list_benchmarks.

    `register(info: BenchmarkDatasetInfo) ‑> None`
    :   Register a new benchmark dataset info.

`BenchmarkTorchDataset(samples: list[dict[str, Any]], tokenizer: Any = None, seq_len: int = 2048, is_causal_lm: bool = True)`
:   PyTorch Dataset wrapper for processed benchmark records.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic
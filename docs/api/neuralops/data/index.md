Module neuralops.data
=====================
Model‑agnostic data pipeline – dataset loading, tokenization, preprocessing,
DataLoader creation, and augmentation for all common tasks and models.

Sub-modules
-----------
* neuralops.data.augmentation
* neuralops.data.base
* neuralops.data.benchmark_datasets
* neuralops.data.cached_tensors
* neuralops.data.custom
* neuralops.data.custom_loader
* neuralops.data.dataset_discovery
* neuralops.data.dataset_loader
* neuralops.data.glue
* neuralops.data.glue_loader
* neuralops.data.multi_choice
* neuralops.data.prepare_cache_once
* neuralops.data.prepare_datasets
* neuralops.data.preprocessing
* neuralops.data.synthetic_generator
* neuralops.data.task_mapper
* neuralops.data.validation

Functions
---------

`convert_to_multichoice_format(df: pandas.DataFrame) ‑> pandas.DataFrame`
:   Normalise a DataFrame into the standard multi‑choice format (question, choices, label).

`create_dataloader(dataset, tokenizer, config: neuralops.data.base.DataConfig, **kwargs) ‑> torch.utils.data.dataloader.DataLoader`
:   Create a DataLoader from a HuggingFace Dataset, applying preprocessing if not already tokenized.
    Uses ProductionDataCollator to guarantee rock-solid batch dimensions and GPU Tensor Core alignment.

`generate_synthetic_data(output_path: str, num_samples: int = 10000, domain: str = 'movies', seed: int = 42) ‑> pandas.DataFrame`
:   Generate a single-domain synthetic classification dataset.

`get_calibration_dataloader(task_name: str = 'sst2', tokenizer: Any = None, batch_size: int = 8, num_samples: int = 128, seq_len: int = 128, split: str = 'train') ‑> Any`
:   Builds a calibration DataLoader from local disk-cached GLUE data.
    Used by data-aware SVD compression algorithms (ASVD, A3, FRanDI, DynamicRank,
    SwiftSVD, SliceGPT, etc.) to collect activation covariance and Fisher information.

`get_metric_name(task: str) ‑> str`
:   Return the metric name.

`get_num_labels(task: str, dataset=None) ‑> int`
:   Return the num labels.

`get_num_labels_and_is_regression(task_name: str) ‑> tuple[int, bool]`
:   Return (num_labels, is_regression) for GLUE tasks.

`get_text_columns(task: str) ‑> list`
:   Return the text columns.

`is_multichoice_dataset(df: pandas.DataFrame) ‑> tuple[bool, int]`
:   Check if a DataFrame contains a multi‑choice format and return the number of choices.

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

`load_dataset(config: neuralops.data.dataset_loader.ExperimentConfig) ‑> datasets.dataset_dict.DatasetDict`
:   Load a dataset based on ExperimentConfig.
    Returns DatasetDict with at least 'train' and optionally 'validation'/'test' splits.

`load_glue_local(task_name: str, base_path: str = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.

`preprocess_function(examples, tokenizer, config: neuralops.data.base.DataConfig) ‑> Any`
:   Generic preprocessing function handling all TaskTypes.
    Returns a dictionary with 'input_ids', 'attention_mask', and appropriate labels.

`random_mask_augment(dataset, mask_prob=0.15) ‑> Any`
:   Random mask augment.

`run_universal_generation(output_dir: str | Path, total_samples_per_domain: int = 5000, domains: dict[str, Any] | None = None, seed: int = 42) ‑> pandas.DataFrame`
:   Generate synthetic dataset across multiple domains and combine them.

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

`DataConfig(task_type: neuralops.data.base.TaskType = TaskType.SINGLE_CLASSIFICATION, task_name: str | None = None, max_length: int = 128, padding: Any = 'max_length', truncation: bool = True, return_tensors: str = 'pt', text_column: str | None = None, text_column_1: str | None = None, text_column_2: str | None = None, label_column: str | None = None, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0, mlm_probability: float = 0.15, question_column: str | None = None, context_column: str | None = None, answer_column: str | None = None, tokens_column: str | None = None, tags_column: str | None = None, custom_preprocess: <class 'collections.abc.Callable'> | None = None, max_samples: int | None = None, split: str | None = None)`
:   DataConfig(task_type: neuralops.data.base.TaskType = <TaskType.SINGLE_CLASSIFICATION: 'single_classification'>, task_name: str | None = None, max_length: int = 128, padding: Any = 'max_length', truncation: bool = True, return_tensors: str = 'pt', text_column: str | None = None, text_column_1: str | None = None, text_column_2: str | None = None, label_column: str | None = None, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0, mlm_probability: float = 0.15, question_column: str | None = None, context_column: str | None = None, answer_column: str | None = None, tokens_column: str | None = None, tags_column: str | None = None, custom_preprocess: collections.abc.Callable | None = None, max_samples: int | None = None, split: str | None = None)

    ### Instance variables

    `answer_column: str | None`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `context_column: str | None`
    :   The type of the None singleton.

    `custom_preprocess: <class 'collections.abc.Callable'> | None`
    :   The type of the None singleton.

    `label_column: str | None`
    :   The type of the None singleton.

    `max_length: int`
    :   The type of the None singleton.

    `max_samples: int | None`
    :   The type of the None singleton.

    `mlm_probability: float`
    :   The type of the None singleton.

    `num_workers: int`
    :   The type of the None singleton.

    `padding: Any`
    :   The type of the None singleton.

    `question_column: str | None`
    :   The type of the None singleton.

    `return_tensors: str`
    :   The type of the None singleton.

    `shuffle: bool`
    :   The type of the None singleton.

    `split: str | None`
    :   The type of the None singleton.

    `tags_column: str | None`
    :   The type of the None singleton.

    `task_name: str | None`
    :   The type of the None singleton.

    `task_type: neuralops.data.base.TaskType`
    :   The type of the None singleton.

    `text_column: str | None`
    :   The type of the None singleton.

    `text_column_1: str | None`
    :   The type of the None singleton.

    `text_column_2: str | None`
    :   The type of the None singleton.

    `tokens_column: str | None`
    :   The type of the None singleton.

    `truncation: bool`
    :   The type of the None singleton.

`DataValidator(schema: dict[str, typing.Any] = None)`
:   

    ### Methods

    `validate(self, dataset: dict[str, numpy.ndarray]) ‑> neuralops.data.validation.ValidationReport`
    :   Validate.

`ExperimentConfig(dataset_name: str, dataset_path: str | None = None, task_type: str = 'single', text_columns: list[str] | None = None, label_column: str | None = None, num_labels: int = 2, metric: str = 'accuracy', max_seq_length: int = 128, split_ratio: float = 0.1, synthetic_samples: int = 5000, glue_base_path: str | None = None, **kwargs)`
:   Configuration for loading a dataset.
    
    Attributes:
        dataset_name: Name of the dataset (GLUE task, synthetic, file path, or HF identifier).
        dataset_path: Path to local dataset file (for custom or multi‑choice).
        task_type: "single", "pair", "multi_choice", "regression".
        text_columns: List of column names containing text (auto‑detected if None).
        label_column: Name of the label column (auto‑detected if None).
        num_labels: Number of classes (for classification).
        metric: Primary metric for evaluation (e.g., "accuracy").
        max_seq_length: Maximum sequence length for tokenization.
        split_ratio: Fraction of data to use for validation (for custom datasets).
        synthetic_samples: Number of synthetic samples to generate (if dataset_name="synthetic").
        glue_base_path: Base directory for local GLUE Parquet files.
        **kwargs: Additional arguments passed to underlying loaders.

`MultiChoiceDataset(df: pandas.DataFrame, tokenizer, max_length: int = 128, question_column: str | None = None, choices_column: str | None = None, answer_column: str | None = None)`
:   Dataset for multi‑choice questions (ARC, HellaSwag, etc.).

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`ProductionDataCollator(pad_token_id: int = 0, label_pad_token_id: int = -100, pad_to_multiple_of: int | None = 8, max_length: int | None = None)`
:   Production-grade data collator for PyTorch DataLoaders.
    Guarantees standard batch dimensions and types for high-performance GPU servers:
    - input_ids: [B, L], torch.long (padded to max_len or multiple of 8 for Tensor Cores)
    - attention_mask: [B, L], torch.long
    - token_type_ids: [B, L], torch.long (if present)
    - labels: [B] (for 1D classification/regression) or [B, L] (for language modeling with -100 ignore index)

`TaskType(*values)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `LANGUAGE_MODELING`
    :   The type of the None singleton.

    `PAIR_CLASSIFICATION`
    :   The type of the None singleton.

    `QA`
    :   The type of the None singleton.

    `REGRESSION`
    :   The type of the None singleton.

    `SEQ2SEQ`
    :   The type of the None singleton.

    `SINGLE_CLASSIFICATION`
    :   The type of the None singleton.

    `TOKEN_CLASSIFICATION`
    :   The type of the None singleton.

`ValidationReport(checks_passed: int, checks_failed: int, issues: list[dict[str, typing.Any]], is_valid: bool)`
:   ValidationReport(checks_passed: int, checks_failed: int, issues: list[dict[str, typing.Any]], is_valid: bool)

    ### Instance variables

    `checks_failed: int`
    :   The type of the None singleton.

    `checks_passed: int`
    :   The type of the None singleton.

    `is_valid: bool`
    :   The type of the None singleton.

    `issues: list[dict[str, typing.Any]]`
    :   The type of the None singleton.
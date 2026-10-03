Module neuralops.data.base
==========================
Base data utilities: configuration, task‑aware preprocessing, DataLoader creation.

Functions
---------

`create_dataloader(dataset, tokenizer, config: neuralops.data.base.DataConfig, **kwargs) ‑> torch.utils.data.dataloader.DataLoader`
:   Create a DataLoader from a HuggingFace Dataset, applying preprocessing if not already tokenized.
    Uses ProductionDataCollator to guarantee rock-solid batch dimensions and GPU Tensor Core alignment.

`preprocess_function(examples, tokenizer, config: neuralops.data.base.DataConfig) ‑> Any`
:   Generic preprocessing function handling all TaskTypes.
    Returns a dictionary with 'input_ids', 'attention_mask', and appropriate labels.

Classes
-------

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
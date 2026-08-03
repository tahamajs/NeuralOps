# `framework/data` — Universal Data Pipeline

The `data` module provides a **model‑agnostic, task‑aware** data pipeline for loading, preprocessing, augmenting, and validating datasets. It supports GLUE, custom files (CSV/JSON/Parquet), synthetic text generation, multi‑choice datasets, and any other dataset available through the HuggingFace Hub. All components are designed to work with **any** HuggingFace tokenizer (BERT, Llama, Gemma, …) and seamlessly connect to the compression and training pipelines of the Efficient Transformer Toolkit.

---

## Overview

| Module | Purpose |
|--------|---------|
| `base.py` | `DataConfig` / `TaskType`, generic `preprocess_function`, `create_dataloader` |
| `dataset_loader.py` | `ExperimentConfig`, unified `load_dataset()` (GLUE, custom, synthetic, hub) |
| `glue_loader.py` | Load GLUE tasks from local Parquet files, metadata keys |
| `task_mapper.py` | Mapping: task → metric, number of labels, text columns |
| `multi_choice.py` | `MultiChoiceDataset` and helpers for ARC, HellaSwag, … |
| `synthetic_generator.py` | Generate synthetic classification datasets across multiple domains |
| `validation.py` | `DataValidator` – missing values, infinities, schema checks |
| `augmentation.py` | Augmentation registry (random mask, back‑translation placeholder) |
| `custom_loader.py` | Load datasets from CSV, JSON, Parquet files with automatic column detection |
| `dataset_discovery.py` | (optional) list available local datasets |

---

## Quick Start

```python
from framework.data import (
    DataConfig, TaskType, preprocess_function, create_dataloader,
    ExperimentConfig, load_dataset, get_num_labels_and_is_regression,
    GLUE_TASK_KEYS
)
from transformers import AutoTokenizer

# 1. Define the data configuration
config = DataConfig(
    task_type=TaskType.SINGLE_CLASSIFICATION,
    text_column_1="sentence",
    label_column="label",
    max_length=128,
    batch_size=16,
    shuffle=True,
)

# 2. Load a GLUE task
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))

# 3. Tokenize and create DataLoader
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_loader = create_dataloader(dataset["train"], tokenizer, config)
```

---

## `base.py` – Core Data Abstractions

### `TaskType`
Enum with common task types:
- `SINGLE_CLASSIFICATION`, `PAIR_CLASSIFICATION`, `REGRESSION`
- `QA`, `TOKEN_CLASSIFICATION`, `LANGUAGE_MODELING`, `SEQ2SEQ`

### `DataConfig`
Dataclass that holds all preprocessing parameters:
- `task_type`, `max_length`, `batch_size`, `shuffle`
- `text_column_1` / `text_column_2` – for single or pair tasks
- `label_column`, `question_column`, `context_column`, `answer_column` (QA)
- `tokens_column`, `tags_column` (token classification)
- `custom_preprocess` – optional user‑supplied function to bypass default logic

### `preprocess_function(examples, tokenizer, config)`
Handles all task types automatically:
- **Classification/Regression**: tokenizes text(s), converts labels to int/float
- **QA**: tokenizes question+context, computes start/end token positions from character offsets
- **Token Classification**: tokenizes words (already split), aligns labels with word IDs
- **Language Modeling**: simple tokenization (labels are handled by the trainer)

### `create_dataloader(dataset, tokenizer, config)`
Applies `preprocess_function` via `dataset.map()`, sets format to PyTorch tensors, returns a standard PyTorch `DataLoader`.

---

## `dataset_loader.py` – Unified Dataset Loading

### `ExperimentConfig`
A single configuration object for loading any dataset. Key fields:
- `dataset_name` – GLUE task name, file path, `"synthetic"`, `"multi_choice"`, or HF hub identifier
- `dataset_path` – path to local file (for custom/multi‑choice)
- `task_type`, `text_columns`, `label_column`, `num_labels`, `metric`
- `split_ratio` – how much data to reserve for validation (custom datasets)
- `synthetic_samples` – number of samples to generate (if `dataset_name="synthetic"`)
- `glue_base_path` – directory containing GLUE Parquet files (auto‑resolved to the framework’s glue dir if not set)

### `load_dataset(config) -> DatasetDict`
Dispatches to the appropriate loader based on `config.dataset_name`:
- **GLUE tasks** → `_load_glue()`
- **`"synthetic"`** → `_load_synthetic()`
- **File extensions** (`.csv`, `.json`, `.parquet`) → `_load_custom()`
- **`"multi_choice"`** → `_load_multi_choice()`
- **Any other string** → HuggingFace `datasets.load_dataset()`

All paths return a HuggingFace `DatasetDict` with at least a `"train"` split.

---

## `glue_loader.py` – GLUE Task Support

- **`GLUE_TASK_KEYS`** – dictionary mapping task name to `(sentence1_key, sentence2_key)`
- **`load_glue_local(task_name, base_path)`** – loads a GLUE task from local Parquet files (supports standard HF cache naming). Handles MNLI’s matched/mismatched splits automatically.
- **`get_num_labels_and_is_regression(task_name)`** – returns `(num_labels, is_regression)`

---

## `task_mapper.py` – Task Metadata

- `TASK_METRICS` – metric name per task (e.g., `"sst2"` → `"accuracy"`, `"stsb"` → `"pearson"`)
- `TASK_NUM_LABELS` – number of classes for classification tasks
- `TASK_COLUMNS` – list of text column names per task
- Functions: `get_metric_name(task)`, `get_num_labels(task, dataset=None)`, `get_text_columns(task)`

---

## `multi_choice.py` – Multi‑Choice Datasets

- **`MultiChoiceDataset(df, tokenizer, max_length, …)`** – PyTorch `Dataset` that returns a stack of tokenized inputs for each choice, suitable for models that score multiple options.
- **`is_multichoice_dataset(df)`** – detect if a DataFrame contains multi‑choice data (based on `choices_json`, `endings`, or `choice_*` columns)
- **`convert_to_multichoice_format(df)`** – normalise any multi‑choice DataFrame into a standard `(question, choices, label)` format

---

## `synthetic_generator.py` – Synthetic Data Generation

- **`generate_synthetic_data(output_path, num_samples, domain, seed)`** – generates a single‑domain classification dataset (positive/negative reviews) using templates and domain‑specific vocabulary. Saves as CSV.
- **`run_universal_generation(output_dir, total_samples_per_domain, domains, seed)`** – generates data across multiple domains (movies, tech, restaurants, books) and combines them into a master CSV.

Domains are defined in `DEFAULT_DOMAINS` with positive/negative adjectives, verbs, nouns, and templates.

---

## `validation.py` – Data Validation

- **`DataValidator(schema)`** – validates a dictionary of numpy arrays or lists for:
  - Missing values
  - Infinite values
  - Zero variance
  - Schema bounds (`min`, `max`)
- Returns a `ValidationReport` with `checks_passed`, `checks_failed`, `issues`, and `is_valid` flag.

---

## `augmentation.py` – Augmentation Registry

- **`AUGMENTATION_REGISTRY`** – a `Registry` instance for data augmentation functions
- **`random_mask_augment(dataset, mask_prob=0.15)`** – replaces random tokens with `[MASK]` (id 103) for BERT‑style masking
- **`dummy_back_translation(dataset)`** – placeholder for future back‑translation augmentation

Register new augmentations using `@AUGMENTATION_REGISTRY.register("name")`.

---

## `custom_loader.py` – Custom File Loading

- **`load_custom_dataset(file_path, text_column, label_column, file_format)`** – loads a CSV/JSON/Parquet file into a `DatasetDict`. Auto‑detects text and label columns if not specified.
- **`load_dataset_from_file(file_path, split_ratio, shuffle, seed)`** – splits a single file into train/validation sets.

---

## `dataset_discovery.py` (optional)

- **`list_local_datasets(base_dir)`** – returns names of subdirectories inside `base_dir`, useful for discovering GLUE datasets stored locally.

---

## Integration with the Rest of the Framework

- **Attention Injection & Compression**: After loading a model and injecting custom attention (using `framework.core.attention_utils.inject_attention()`), you can directly use the data pipeline to prepare DataLoaders for fine‑tuning or evaluation.
- **Training Script**: Use `DataConfig` together with `TrainingArguments` from HuggingFace. The `create_dataloader` output is ready for `Trainer`.
- **Full Example**:

```python
from framework.core import UnifiedTransformerModel, compress_model
from framework.data import load_dataset, DataConfig, TaskType, create_dataloader
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Model
model = UnifiedTransformerModel.from_pretrained("distilbert-base-uncased")
model.replace_attention("galerkin", rank=32)

# Data
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))
config = DataConfig(task_type=TaskType.SINGLE_CLASSIFICATION,
                    text_column_1="sentence", label_column="label")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_loader = create_dataloader(dataset["train"], tokenizer, config)
val_loader = create_dataloader(dataset["validation"], tokenizer, config)

# Train
training_args = TrainingArguments(...)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_loader.dataset,
                  eval_dataset=val_loader.dataset)
trainer.train()
```

---

## Supported Models & Tokenizers

The data pipeline works with **any** HuggingFace tokenizer. The preprocessing function uses standard tokenizer methods (`tokenizer()`, `tokenizer.word_ids()`) and is thus independent of model architecture. Tested with:

- BERT, DistilBERT, RoBERTa
- GPT‑2
- Llama‑2, Llama‑3, Mistral, Gemma, Phi

---

## Extending the Data Pipeline

- **New task types**: Add an entry to `TaskType` and a corresponding branch in `preprocess_function`.
- **New dataset source**: Implement a loader function and add a dispatch in `load_dataset`.
- **Custom preprocessing**: Provide a `custom_preprocess` callable in `DataConfig`.
- **New augmentations**: Decorate a function with `@AUGMENTATION_REGISTRY.register(...)`.

The data module is designed to be as generic as possible—any model, any task, any data source.
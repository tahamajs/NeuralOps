# 📦 Data Module

A unified dataset loading and preprocessing toolkit for the Efficient Transformer Toolkit.  
Supports GLUE benchmarks, custom files (CSV/JSON/Parquet), multi‑choice datasets, and synthetic data generation.

## 📁 Structure

```
framework/data/
├── __init__.py                # Consolidated exports
├── dataset_loader.py          # Unified loader (ExperimentConfig + load_dataset)
├── glue_loader.py             # Local GLUE Parquet loader
├── custom_loader.py           # CSV/JSON/Parquet loader with auto‑detection
├── multi_choice.py            # MultiChoiceDataset and utilities
├── preprocessing.py           # Tokenization helpers
├── synthetic_generator.py     # Synthetic text classification data
├── task_mapper.py             # GLUE metadata (metrics, labels, columns)
└── README.md                  # This file
```

## 🚀 Quick Start

```python
from framework.data import load_dataset, ExperimentConfig

# 1. Load a GLUE task
config = ExperimentConfig(
    dataset_name="mrpc",
    glue_base_path="/path/to/glue"
)
dataset = load_dataset(config)

# 2. Load a custom CSV file
config = ExperimentConfig(
    dataset_name="my_data.csv",
    split_ratio=0.8
)
dataset = load_dataset(config)

# 3. Generate synthetic data
config = ExperimentConfig(
    dataset_name="synthetic",
    synthetic_samples=2000
)
dataset = load_dataset(config)
```

## 📚 Detailed Documentation

### 1. `ExperimentConfig` (Configuration)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `dataset_name` | str | GLUE task, file path, "synthetic", or HF hub name | required |
| `dataset_path` | str | Path to custom dataset file (if not in `dataset_name`) | None |
| `task_type` | str | `"single"`, `"pair"`, `"multi_choice"`, `"regression"` | `"single"` |
| `text_columns` | list | Column names for text (auto‑detected if None) | None |
| `label_column` | str | Column name for labels (auto‑detected) | None |
| `num_labels` | int | Number of classes (classification) | 2 |
| `metric` | str | Evaluation metric | `"accuracy"` |
| `max_seq_length` | int | Tokenization max length | 128 |
| `split_ratio` | float | Fraction for validation (custom datasets) | 0.1 |
| `synthetic_samples` | int | Number of samples (synthetic) | 5000 |
| `glue_base_path` | str | Base directory for GLUE Parquet files | `"./data/glue"` |

### 2. `load_dataset()` – Unified Loader

```python
def load_dataset(config: ExperimentConfig) -> DatasetDict
```

Returns a Hugging Face `DatasetDict` with at least `"train"` and (if applicable) `"validation"` splits.

**Supported dataset types** (auto‑detected via `dataset_name`):

| Type | Name / Pattern | Behaviour |
|------|----------------|-----------|
| GLUE | `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `mnli`, `qnli`, `rte`, `wnli` | Loads local Parquet files from `glue_base_path`. |
| Synthetic | `"synthetic"` | Generates synthetic data with `synthetic_samples`. |
| Custom file | ends with `.csv`, `.json`, `.jsonl`, `.parquet` | Loads file; auto‑splits train/val using `split_ratio`. |
| Multi‑choice | `"multi_choice"` | Loads raw dataset (use `MultiChoiceDataset` afterwards). |
| Hugging Face hub | any other string | Downloads from `datasets` library; adds validation split if needed. |

### 3. GLUE Loader (`glue_loader.py`)

Handles local Parquet files named `train.parquet`, `validation.parquet`, `test.parquet`.  
Special handling for MNLI (`validation_matched`, `validation_mismatched`).

```python
from framework.data import load_glue_local, get_num_labels_and_is_regression

dataset = load_glue_local("mrpc", "/path/to/glue")
num_labels, is_regression = get_num_labels_and_is_regression("mrpc")
```

### 4. Custom Loader (`custom_loader.py`)

```python
from framework.data import load_custom_dataset, load_dataset_from_file

# Load without splitting
dataset = load_custom_dataset("data.csv", text_column="text", label_column="label")

# Load and split automatically
dataset = load_dataset_from_file("data.csv", split_ratio=0.8)
```

**Auto‑detection**: if `text_column`/`label_column` not provided, the loader attempts to find the most suitable column by heuristics.

### 5. Multi‑Choice Datasets (`multi_choice.py`)

```python
from framework.data import MultiChoiceDataset, is_multichoice_dataset

df = pd.read_csv("arc.csv")
is_mc, num_choices = is_multichoice_dataset(df)   # (True, 4)
dataset = MultiChoiceDataset(df, tokenizer, max_length=128)
batch = dataset[0]  # returns {'input_ids': (num_choices, seq_len), ...}
```

**Supported formats**:
- Columns: `choices_json` (JSON list), `endings` (JSON list), or `choice_a`, `choice_b`, … columns.
- Answer can be in `gold_answer_json` (with `choice_index`), `answer_json`, or a simple `label` column.

### 6. Preprocessing (`preprocessing.py`)

```python
from framework.data import preprocess_function

preprocess = preprocess_function(tokenizer, task_name="mrpc", max_seq_length=128)
encoded_dataset = dataset.map(preprocess, batched=True)
```

For multi‑choice, use `MultiChoiceDataset` directly; the preprocessing function simply returns the examples unchanged.

### 7. Synthetic Generator (`synthetic_generator.py`)

```python
from framework.data import run_universal_generation, generate_synthetic_data

# Multi‑domain generation (movies, tech, restaurants, books)
df = run_universal_generation(output_dir=Path("./synthetic"), total_samples_per_domain=5000)

# Single‑domain generation
df = generate_synthetic_data("output.csv", num_samples=10000, domain="movies")
```

The generator produces balanced positive/negative samples using configurable templates and vocabulary.

### 8. Task Mapper (`task_mapper.py`)

```python
from framework.data import get_metric_name, get_num_labels, get_text_columns

metric = get_metric_name("cola")          # "matthews_correlation"
num_labels = get_num_labels("mrpc")       # 2
text_cols = get_text_columns("mnli")      # ["premise", "hypothesis"]
```

Useful for dynamic task handling in training scripts.

## 🔧 Integration with the Training Pipeline

```python
from framework.data import load_dataset, ExperimentConfig, preprocess_function
from transformers import AutoTokenizer

config = ExperimentConfig(dataset_name="mrpc", glue_base_path="./data/glue")
raw_dataset = load_dataset(config)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
preprocess = preprocess_function(tokenizer, task_name="mrpc")
encoded_dataset = raw_dataset.map(preprocess, batched=True)
```

## 📜 Dependencies

- `datasets` (Hugging Face)
- `pandas`, `numpy`
- `torch`
- `scikit‑learn` (for metrics, used internally)
- `pyarrow` (for Parquet support)

## 📄 License

Part of the **Efficient Transformer Toolkit** – MIT License.


# 📊 Data Loading Pipeline

The `framework.data` package provides a **unified interface** to load any dataset – from GLUE benchmarks to custom CSVs, multi‑choice exams, and synthetic generation.

## 🎯 Key Features

- **One‑line loading** – `load_dataset(config)` reads an `ExperimentConfig` and returns a ready‑to‑tokenize `DatasetDict`.
- **Auto‑split** – custom datasets are automatically split into train/validation.
- **Multi‑task support** – single‑sentence, sentence‑pair, regression, multi‑choice.
- **Extensible** – easily add new tasks by adding a loader function.

## 🚀 Quick Start

```python
from framework.data import load_dataset, ExperimentConfig

config = ExperimentConfig(
    dataset_name="mrpc",         # GLUE task
    max_seq_length=128,
    glue_base_path="./data/glue"
)

dataset = load_dataset(config)
print(dataset.keys())  # dict_keys(['train', 'validation', 'test'])
```

## ⚙️ Configuration

Create an `ExperimentConfig` with these fields:

| Parameter | Description |
|-----------|-------------|
| `dataset_name` | GLUE task name, file path (.csv/.json/etc), or `"synthetic"` |
| `dataset_path` | Optional explicit path to file |
| `task_type` | `"single"`, `"pair"`, `"multi_choice"`, `"regression"` |
| `text_columns` | List of column names for text (e.g. `["sentence1","sentence2"]`) |
| `label_column` | Label column name |
| `num_labels` | Number of output classes |
| `metric` | Evaluation metric (`"accuracy"`, `"matthews_correlation"`, etc.) |
| `max_seq_length` | Tokenization max length |
| `split_ratio` | Fraction for validation when no validation split exists |
| `synthetic_samples` | Number of samples per domain for synthetic generation |
| `glue_base_path` | Path to folder containing GLUE parquet files |

## 📦 Supported Dataset Types

### 1. GLUE Benchmarks
- All standard tasks: CoLA, SST‑2, MRPC, QQP, STSB, MNLI, QNLI, RTE, WNLI.
- Loaded from local Parquet files (expected structure: `{task_name}/train-00000-of-00001.parquet` etc.).

### 2. Custom CSV / JSON / Parquet
- Any file with text and optional label columns.
- Columns auto‑detected.
- Use `split_ratio` to create a validation split.

### 3. Multi‑choice Datasets
- Supports ARC, HellaSwag, and similar.
- Converts `choices_json` or `endings` into tokenized input tensors.

### 4. Synthetic Data
- Generates domain‑specific classification datasets using templates.
- Domains: `movies`, `tech`, `restaurants`, `books`.
- Useful for quick prototyping.

## 🧠 Integration with Training

After loading, tokenize with the provided `preprocess_function`:

```python
from framework.data import preprocess_function

fn = preprocess_function(tokenizer, task_name="mrpc", max_seq_length=128)
dataset = dataset.map(fn, batched=True)
```

For multi‑choice, use `MultiChoiceDataset` directly:

```python
from framework.data import MultiChoiceDataset
from torch.utils.data import DataLoader

train_dataset = MultiChoiceDataset(train_df, tokenizer, max_length=128)
loader = DataLoader(train_dataset, batch_size=8)
```

## 🛠️ Adding a New Dataset Type

1. Add a new `_load_xxx` function in `dataset_loader.py`.
2. Register it in `load_dataset` by name or extension.
3. Optionally add preprocessing logic in `preprocessing.py`.

## 📂 Folder Structure

```
framework/data/
├── __init__.py
├── dataset_loader.py
├── custom_loader.py
├── glue_loader.py
├── multi_choice.py
├── preprocessing.py
├── synthetic_generator.py
├── task_mapper.py
└── README.md
```

---

*Load any dataset, any format, with one line of configuration.*
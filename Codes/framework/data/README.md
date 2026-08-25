# Data Subsystem (`Codes/framework/data`)

The `data` subsystem provides a unified dataset ingestion, tokenization, preprocessing, multi-task discovery, and synthetic generation pipeline.

---

## 🏗 Architecture Overview

- **GLUE Benchmark Loader (`glue_loader.py`, `glue.py`)**: Native support for standard GLUE tasks (MRPC, SST-2, QNLI, RTE, MNLI, QQP, STS-B, CoLA) with automatic caching and label normalization.
- **Dataset Preprocessing & Validation (`prepare_datasets.py`, `validation.py`)**: Token length filtering, dynamic padding, truncation, sequence packing, and schema integrity validation.
- **Multi-Choice Classification (`multi_choice.py`)**: Flattening and batch collation for multiple-choice reasoning datasets (e.g. SWAG, ARC, HellaSwag).
- **Synthetic Data Generator (`synthetic_generator.py`)**: Deterministic synthetic token and sequence generator for offline CI/CD pipelines, quick smoke tests, and kernel profiling.
- **Augmentation & Task Mapping (`augmentation.py`, `task_mapper.py`)**: Text perturbation, back-translation hooks, and task metadata mapping.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `glue_loader.py` | High-level loader for GLUE benchmark datasets with offline caching. |
| `prepare_datasets.py` | Tokenization pipeline converting raw HuggingFace datasets to model inputs. |
| `dataset_loader.py` | Base dataset loading utilities and DataLoader factory helpers. |
| `multi_choice.py` | Specialized preprocessor and collator for multiple-choice QA tasks. |
| `synthetic_generator.py` | Offline synthetic dataset generator for zero-network testing. |
| `validation.py` | Pre-flight dataset verification and schema validator. |

---

## 🚀 Quick Start & Usage

### 1. Loading GLUE Benchmark Datasets
```python
from framework.data.glue_loader import load_glue_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_ds, eval_ds = load_glue_dataset(
    task_name="mrpc",
    tokenizer=tokenizer,
    max_length=128,
    batch_size=32,
)
```

### 2. Generating Synthetic Dataset for Offline Tests
```python
from framework.data.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(vocab_size=1000, seq_len=64)
dataset = generator.generate_classification_dataset(num_samples=100, num_classes=2)
```

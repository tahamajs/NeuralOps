# Data Pipeline Subsystem (`Codes/framework/data`)

The `data` subsystem provides an end-to-end dataset preparation, caching, and loading pipeline for benchmark evaluation and training on CPU and GPU environments.

---

## 🏗 Architecture & Local Cache Structure

```
Codes/framework/data/
├── glue_data/              # Pre-cached GLUE dataset splits on local disk
│   ├── sst2/               # SST-2 (Stanford Sentiment Treebank)
│   ├── mrpc/               # MRPC (Microsoft Research Paraphrase Corpus)
│   ├── qnli/               # QNLI (Question Natural Language Inference)
│   ├── cola/               # CoLA (Corpus of Linguistic Acceptability)
│   ├── rte/                # RTE (Recognizing Textual Entailment)
│   ├── qqp/                # QQP (Quora Question Pairs)
│   ├── mnli/               # MNLI (Multi-Genre Natural Language Inference)
│   ├── stsb/               # STS-B (Semantic Textual Similarity Benchmark)
│   └── wnli/               # WNLI (Winograd NLI)
├── prepare_datasets.py     # Master dataset preparation script (HF Hub -> disk)
├── glue_loader.py          # Unified GLUE loader (local disk cache first, then HF Hub)
├── benchmark_datasets.py   # Benchmark batch collators and schema formatters
└── synthetic_generator.py  # Deterministic synthetic data generator for offline testing
```

---

## 🚀 Usage Guide

### 1. `load_glue_split()` vs `prepare_glue_dataset()`
- **`prepare_glue_dataset()`**: Downloads raw splits from HF Hub, applies tokenizer, and persists tokenized tensors or Arrow tables to disk.
- **`load_glue_split()`**: First inspects `glue_data/<task>` on local disk for instant offline zero-download loading; falls back to HF Hub if absent, then applies tokenizer and formats tensors.

### 2. Dataset Preparation via CLI
```bash
# Prepare specific tasks with custom tokenizers
python prepare_datasets.py --tasks sst2,mrpc --models bert,roberta,t5 --save_dir ./prepared_data

# Prepare all GLUE and LM benchmarks in batch
python prepare_datasets.py --all --models bert,roberta,t5,distilbert --save_dir /mnt/data
```

### 3. Loading in Python
```python
from framework.data.glue_loader import load_glue_split

# Automatically strips token_type_ids for T5 and handles padding for GPT-2
val_dataset = load_glue_split(
    task="sst2",
    split="validation",
    tokenizer_name="bert-base-uncased",
    max_seq_len=128,
)
```

---

## ⚙️ Tokenizer Compatibility Guarantees

| Architecture | Tokenizer | Special Handling |
|:---|:---|:---|
| **BERT / DistilBERT** | `bert-base-uncased` | Standard `input_ids`, `attention_mask`, `token_type_ids` |
| **RoBERTa** | `roberta-base` | BPE tokenization, byte-level vocabulary (50,265 tokens) |
| **T5** | `t5-small` | `token_type_ids` strictly stripped from tensors |
| **GPT-2** | `gpt2` | Missing `pad_token` set to `eos_token` automatically |

---

## 🖥️ GPU Server Bootstrap One-Liner

```bash
python3 Codes/framework/data/prepare_datasets.py --all --models bert,roberta,t5,distilbert --save_dir ./prepared_benchmark_data
```

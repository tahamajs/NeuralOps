# 📊 Consolidated Datasets & Automated Preprocessing Hub

All project datasets are consolidated into the central `/datasets` directory, supported by the automated tokenization and streaming pipeline in `Codes/framework/data/prepare_datasets.py`.

---

## 📁 Folder Structure

```
datasets/
├── cache/                  # Serialized .pt tokenized batches for zero-latency DataLoader streaming
│   ├── glue_sst2_bert-base-uncased.pt
│   ├── glue_mrpc_roberta-base.pt
│   └── lm_wikitext2_t5-base.pt
├── glue/                   # Standard GLUE benchmark parquet/arrow splits (SST-2, MRPC, QNLI, MNLI, etc.)
├── coco/                   # COCO dataset files and visual grounding annotations
├── udt-coco/               # Unified Data Tool annotations
├── bigbench/               # BIG-bench multi-task prompting & evaluation suites
├── truthful_qa/            # TruthfulQA multiple choice and generation splits
└── synthetic/              # High-fidelity deterministic synthetic evaluation benchmarks
```

---

## ⚡ Automated Dataset Preparation CLI

Pre-download, tokenize, dynamically pad, and cache all datasets for air-gapped GPU servers:

```bash
# Tokenize and cache GLUE + LM benchmarks for BERT, RoBERTa, and T5
PYTHONPATH=Codes python3 Codes/framework/data/prepare_datasets.py \
  --benchmarks glue,lm \
  --models bert-base-uncased,roberta-base,t5-base \
  --output_dir datasets/cache/ \
  --max_seq_len 128 \
  --num_workers 4
```

### Preprocessing Features:
1. **Zero-Latency Streaming**: Pre-tokenized PyTorch tensors (`input_ids`, `attention_mask`, `labels`) loaded directly to GPU memory with pinned memory support.
2. **Deterministic Offline Synthetic Fallback**: If network is unavailable or HuggingFace is air-gapped, `SyntheticDataset` automatically generates reproducible synthetic batches matching exact vocabularies and sequence distributions.
3. **Multi-Model Tokenizer Mapping**: Correctly maps tokenizers for encoder-only (`bert`, `roberta`), sequence-to-sequence (`t5`), and causal LM (`gpt2`, `llama`).

---

## 🛠️ Usage in Python

```python
from framework.data.prepare_datasets import get_dataset, create_benchmark_dataloader

# Load cached or dynamically preprocessed dataloader
dataloader = get_dataset(
    dataset_name="sst2",
    tokenizer_name="bert-base-uncased",
    split="validation",
    batch_size=32,
    max_seq_len=128
)

for batch in dataloader:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    break
```

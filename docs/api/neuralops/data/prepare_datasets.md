Module neuralops.data.prepare_datasets
======================================
Dataset Preparation & Preprocessing Pipeline for GPU Server Benchmark & Training.

Supports:
- GLUE Suite (SST-2, MRPC, QNLI, MNLI, CoLA, RTE, STS-B, QQP)
- Language Modeling (WikiText-2, WikiText-103, C4, PTB)
- Question Answering & Reasoning (SQuAD v2, GSM8K subset, MMLU subset)
- Fast deterministic synthetic generation when offline / air-gapped
- Multi-model tokenization & dynamic padding (BERT, RoBERTa, T5, GPT-2, LLaMA)
- PyTorch DataLoader serialization to disk (.pt / Arrow) for instant GPU execution

Usage:
  # Prepare GLUE datasets with default tokenizers
  python prepare_datasets.py --tasks sst2,mrpc,qnli,cola --save_dir ./glue_data

  # Prepare all benchmark datasets for GPU server
  python prepare_datasets.py --all --save_dir ./benchmark_data --batch_size 256

  # Test dataset loading and print batch schemas
  python prepare_datasets.py --test --tasks sst2,wikitext2

Functions
---------

`create_benchmark_dataloader(dataset: Dataset, batch_size: int = 16, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, device: torch.device | None = None) ‑> torch.utils.data.dataloader.DataLoader`
:   Creates a high-throughput PyTorch DataLoader configured for GPU training/benchmarking.

`main() ‑> None`
:   

`prepare_all_benchmarks(save_dir: str = './benchmark_data', tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128) ‑> dict[str, dict[str, torch.utils.data.dataset.Dataset]]`
:   Prepares and caches all GLUE and LM benchmarks.

`prepare_glue_dataset(task_name: str, tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128, save_dir: str | None = None, use_synthetic_fallback: bool = True) ‑> dict[str, torch.utils.data.dataset.Dataset]`
:   Downloads, tokenizes, and formats a GLUE benchmark task.
    Saves tokenized dataset tensors to disk if save_dir is specified.

`prepare_lm_dataset(dataset_name: str = 'wikitext2', tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128, save_dir: str | None = None) ‑> dict[str, torch.utils.data.dataset.Dataset]`
:   Prepares Language Modeling benchmarks (WikiText-2, C4).

Classes
-------

`SyntheticDataset(task_name: str, num_samples: int = 1000, seq_len: int = 64, vocab_size: int = 30522, num_labels: int = 2, seed: int = 42)`
:   Deterministic synthetic in-memory dataset for air-gapped or offline GPU execution.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic
Module neuralops.benchmarks.run_long_context_sweep
==================================================
Long-Context Sweep Runner: Evaluates compression modules on Byte-Level IMDb (4096 tokens).

Evaluates:
  - Sequence length: 4096 bytes
  - Batch size: 2 (safely fits 8 GB GPU VRAM)
  - Metrics: Classification Accuracy (%), Peak VRAM (MB), Latency (ms/sample), Compression Rate

Usage:
    python run_long_context_sweep.py --dry_run
    python run_long_context_sweep.py --model_path models/distilbert_sst2 --rank 64

Functions
---------

`adapt_model_pos_embeddings(model: nn.Module, target_seq_len: int = 4096) ‑> None`
:   Expands position embeddings to target_seq_len via linear interpolation if needed.

`evaluate_long_context(model, dataloader, max_samples=None, device='cuda')`
:   Evaluates sequence classification on long context with VRAM and latency tracking.

`get_byte_imdb_dataloader(split: str = 'test', batch_size: int = 8, max_length: int = 4096, dataset_dir: str = 'data/long_context/byte_imdb') ‑> Any`
:   Loads or creates a Byte-level IMDb DataLoader for long context evaluations.

`main()`
:   

`parse_args()`
:
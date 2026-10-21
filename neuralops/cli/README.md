# NeuralOps Command-Line Interface (CLI) Suite

Command-line interface for the NeuralOps deep learning and attention optimization framework (v1.5.0).

## Canonical Inventory
- **Attention Mechanisms:** 131 registered modules across 11 mathematical families
- **Compression Engines:** 72 registered algorithms across 7 mathematical families
- **Triton Kernel Modules:** 26 specialized JIT-compiled kernel operators
- **CUDA Kernel Implementations:** 114 accelerated C++/CUDA kernel implementations
- **Framework Version:** 1.5.0

---

## Command Reference

### 1. `neuralops train`
Pretrain or fine-tune models with multi-corpus streaming, variable-length document packing, custom schedules, and advanced optimizers.

```bash
# Fine-tune model with standard datasets
neuralops train --model gpt2 --dataset wikitext --epochs 3 --batch-size 8 --lr 1e-4

# Pretrain on multi-corpus stream with variable-length document packing (cu_seqlens)
neuralops train \
  --model distilbert-base-uncased \
  --dataset "HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2" \
  --varlen \
  --seq-len 65536 \
  --context-len 4096 \
  --optimizer lion \
  --lr-scheduler wsd \
  --decay-ratio 0.2 \
  --save-dir results/trained_model \
  --compile
```

**Key Arguments:**
- `--model`: Hugging Face identifier or local checkpoint path.
- `--dataset`: Target dataset or comma-separated list of multi-datasets.
- `--dataset-name` / `--subset`: Dataset subset or configuration name.
- `--split`: Target split (default: `train`).
- `--varlen`: Enable zero-padding variable-length document packing.
- `--seq-len`: Target packed sequence length (e.g. 4096, 65536).
- `--context-len`: Maximum individual document sample length (default: 4096).
- `--batch-size`: Batch size per device.
- `--gradient-accumulation-steps`: Gradient accumulation steps.
- `--optimizer`: `adamw`, `lion`, `muon`, or `sgd`.
- `--lr-scheduler`: `wsd` (Warmup-Stable-Decay), `cosine`, or `linear`.
- `--decay-ratio`: Ratio of total steps allocated to cooldown decay phase in WSD schedule (default: 0.2).
- `--compile`: JIT-compile model with `torch.compile()`.

---

### 2. `neuralops eval`
Comprehensive model evaluation measuring perplexity, token throughput, latency, and peak VRAM.

```bash
# Standard evaluation
neuralops eval --model gpt2 --dataset wikitext --batch-size 16 --max-samples 500

# JSON output to file for automated benchmarking pipelines
neuralops eval --model gpt2 --dataset sst2 --metrics perplexity,throughput,latency,peak_vram --output-file results/eval.json
```

**Key Arguments:**
- `--model`: Target model identifier or checkpoint path.
- `--dataset`: Evaluation dataset (e.g. `wikitext`, `sst2`, `c4`).
- `--batch-size`: Evaluation batch size.
- `--max-samples`: Maximum samples to evaluate.
- `--device`: Target device (`cuda`, `mps`, `cpu`).
- `--metrics`: Comma-separated list of metrics.
- `--json` / `--output-json`: Print results formatted as JSON.
- `--output-file`: Destination path to save JSON results.

---

### 3. `neuralops compress`
Compress model weights via Singular Value Decomposition (SVD), low-rank factorization, quantization, and structured pruning.

```bash
# Direct SVD compression to a target rank
neuralops compress --model gpt2 --compressor direct_svd --rank 32 --output-dir results/compressed_gpt2

# Select a rank fraction for the direct-SVD family
neuralops compress --model gpt2 --compressor direct_svd --target-ratio 0.5 --output-dir results/compressed_svd
```

**Key Arguments:**
- `--model`: Base model identifier or path.
- `--compressor`: Registered algorithm (for example, `direct_svd`, `rpca_compressor`, or `tt_compressor`).
- `--rank`: Target rank for low-rank factorization.
- `--target-ratio`: Rank fraction for `direct_svd` and its aliases; it does not guarantee the same model-wide parameter reduction. Other compressors reject this option rather than silently ignoring it.
- `--output-dir`: Output directory for compressed weights and manifest.
- `--json`: Output compression metrics as JSON.

---

### 4. `neuralops serve`
Launch a basic OpenAI-compatible HTTP API backed by a pretrained Hugging Face causal language model. This command uses ordinary `generate()` calls; continuous batching, PagedKVCache, and speculative decoding are not wired into this CLI server.

```bash
# Start the HTTP server
neuralops serve --model gpt2 --host 127.0.0.1 --port 8000
```

**Key Arguments:**
- `--model`: Primary model identifier or local checkpoint path.
- `--host`: Network interface to bind (default: `0.0.0.0`).
- `--port`: Network port to listen on (default: `8000`).
- `--engine`: Currently implemented engine: `fastapi`.

The serving adapter requires FastAPI, Uvicorn, and Transformers. Use vLLM or another dedicated serving engine when continuous batching or production scheduling is required.

Install the optional server dependencies with `pip install 'neuralops[serving]'`.

---

### 5. `neuralops test`
Fast diagnostic test runner verifying environment health, kernel capabilities, and attention surgery integrity.

```bash
# Run all diagnostic suites
neuralops test --all

# Run specific diagnostic suites
neuralops test --health
neuralops test --kernels
neuralops test --surgery
```

**Key Arguments:**
- `--all`: Run all diagnostic suites.
- `--health`: Check environment, dependencies, and registry integrity.
- `--kernels`: Verify 26 Triton kernel modules and 114 CUDA kernel implementations.
- `--surgery`: Verify attention layer surgery and alpha blending.
- `-q`, `--quiet`: Quiet mode reporting only pass/fail status.

---

### 6. `neuralops list`
Display registered attention mechanisms and compression engines categorized by mathematical family and hardware backend support.

```bash
# List all 131 attention mechanisms and 72 compression engines
neuralops list

# Filter by category or backend
neuralops list --type attention --backend triton
neuralops list --type compressor --backend cuda

# Output raw JSON catalog
neuralops list --json
```

**Key Arguments:**
- `--type`: `attention`, `compressor`, `kernels`, or `all`.
- `--family`: Substring filter for mathematical family.
- `--backend`: Hardware backend filter (`cuda`, `triton`, `pytorch`, `all`).
- `--json`: Output catalog formatted as JSON.

---

### 7. `neuralops info`
Display comprehensive environment, GPU status, and canonical inventory summary.

```bash
neuralops info
```

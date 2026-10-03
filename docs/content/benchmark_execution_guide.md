# 🧪 NeuralOps Benchmark Execution Guide & Cluster Runner

> **Standard**: Publication-Grade Empirical Benchmarking Protocol  
> **Evaluation Tracks**: SVD & Low-Rank, Spectral & SSM, MoE & Routing, Kernel & Attention Approximations, Structural Pruning & Quantization  
> **Backbones**: BERT-Base, RoBERTa-Base, DistilBERT, T5-Base, LLaMA-2/3 (7B/13B), Mistral-7B

---

## 1. Overview & Reproducibility Mission

The **NeuralOps Framework** provides unified benchmarking across **236 Attention Mechanisms** and **107 Compression Engines**. To maintain strict scientific integrity for conference submissions (NeurIPS / ICLR / ICML / ACL):

1. **Real Dataset Pipelines**: Benchmarks evaluate on authentic GLUE splits (SST-2, MRPC, QNLI, MNLI, RTE, CoLA) and Long-Sequence benchmarks (LongBench, WikiText-103, C4) using tokenizer-aligned pipelines (`data/glue_loader.py`, `data/prepare_datasets.py`).
2. **Deterministic Multi-Seed Runs**: Every reported number must specify random seeds (standard: $s \in \{42, 1337, 2026\}$) and report mean $\pm$ standard deviation.
3. **Hardware Isolation**: Latency (ms/token) and peak VRAM (MB) metrics must be captured on isolated GPUs with locked clocks to prevent thermal throttling skew.

---

## 2. Command-Line Execution Interface

### 2.1 Single Module Evaluation on GLUE
To evaluate a single attention mechanism or compressed model against a GLUE task:

```bash
# Evaluate SVD Linear Attention on SST-2 using BERT-Base
python Codes/framework/eval_all_models_datasets.py \
    --model bert-base-uncased \
    --dataset sst2 \
    --attn_type svd_linear_attention \
    --rank 32 \
    --batch_size 32 \
    --device cuda

# Evaluate Activation-Aware SVD (AASVD) Compressor on MRPC
python Codes/framework/eval_all_models_datasets.py \
    --model roberta-base \
    --dataset mrpc \
    --compressor aa_svd_compressor \
    --rank 64 \
    --batch_size 16 \
    --device cuda
```

### 2.2 Comprehensive Multi-Track Cluster Execution
To launch the full batch sweep across all registered attention variants and compression engines on a GPU server:

```bash
# Execute full comprehensive validation suite
bash run_on_gpu_server.sh --mode comprehensive --gpus 0,1,2,3 --batch_size 32

# Execute targeted track evaluation (e.g., Track 1: SVD & Low-Rank)
python Codes/framework/eval_all_models_datasets.py \
    --track track1_svd_lowrank \
    --dataset sst2,mrpc \
    --output_dir Codes/framework/benchmark_results/
```

### 2.3 Diagnostic Quick-Check (CPU / Single GPU)
For rapid verification of forward pass, gradient flow, and parameter shape invariants:

```bash
# Test attention replacement and backward pass
python Codes/framework/check_framework_sync.py

# Run targeted unit tests
pytest Codes/framework/tests/test_attention_modules.py -v
pytest Codes/framework/tests/test_compression_engine.py -v
```

---

## 3. Evaluated Metrics & Statistical Reporting

| Metric | Target Dimension | Methodology |
| :--- | :--- | :--- |
| **Accuracy / F1** | Downstream task quality | Evaluated on validation split with standard evaluation metric (`accuracy` for SST-2/QNLI, `f1` and `accuracy` for MRPC). |
| **Perplexity (PPL)** | Language modeling fidelity | $\exp\left(rac{1}{N}\sum_{i=1}^N \mathcal{L}_iight)$ computed over sliding window context of 2048 tokens on WikiText-2 / C4. |
| **Effective Latency** | Inference speed | Warmup: 50 batches; Measurement: 200 batches. Median time per token in milliseconds ($ms/	ext{tok}$). |
| **Peak VRAM Footprint** | Memory consumption | Tracked via `torch.cuda.max_memory_allocated()`. Measured in MB. |
| **Parameter Reduction** | Model compactness | $rac{	ext{Params}_{	ext{orig}} - 	ext{Params}_{	ext{compressed}}}{	ext{Params}_{	ext{orig}}} 	imes 100\%$. |
| **Condition Number $\kappa(W)$** | Numerical stability | $rac{\sigma_{\max}}{\sigma_{\min}}$ of projection matrices across layers to detect gradient stagnation. |

---

## 4. Benchmark Result Storage & Verification Artifacts

All benchmark executions output structured JSON, Parquet, and Markdown summaries into:
- Raw metrics: `Codes/framework/benchmark_results/json/`
- Aggregated CSV summaries: `Codes/framework/benchmark_results/master_summary.csv`
- Pareto frontier curves: `Codes/framework/benchmark_results/pareto_frontier.json`

Researchers should inspect these generated artifacts rather than static mock tables when reporting published comparisons.

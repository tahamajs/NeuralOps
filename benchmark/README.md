# 🧪 Multi-Model Attention Benchmark Framework

A comprehensive, modular, and fault-tolerant benchmarking suite designed to evaluate alternative attention mechanisms registered in `ATTENTION_REGISTRY` on top of HuggingFace transformer backbones (such as `distilbert-base-uncased`).

---

## 🌟 Key Features

1. **Layer 0 (Pre-Flight Sanity Checks)**:
   - Dynamic shape verification (`[B, L, D]`) and dtype preservation.
   - Non-NaN, non-infinite gradient backward flow tests across all trainable parameters.
   - Pre-training parameter dimension compatibility.

2. **Layer 1 (GLUE Task Evaluation)**:
   - Full evaluation across multiple GLUE benchmarks: `SST-2`, `MRPC`, `QNLI`, `RTE`, `CoLA`, and `STS-B`.
   - Automatic early-stopping on majority-class collapse ($Loss > 0.693$ on binary classification).
   - Mixed-precision acceleration (`BF16` / `FP16`) and dynamic batch padding.

3. **Layer 2 (Computational Profiling & Scaling)**:
   - Inference throughput (tokens / second) across sequence lengths `[128, 256, 512]` and batch sizes `[1, 32]`.
   - Peak GPU VRAM footprint tracking and automated Out-Of-Memory (`OOM`) recovery.
   - FLOP estimation using `fvcore` / analytical heuristics.

4. **Layer 3 (Multi-Seed & Hyperparameter Stability)**:
   - Seed sensitivity variance across seeds `[42, 123, 777]`.
   - Learning rate robustness coefficient of variation ($CV_{LR}$).
   - Batch size scaling stability ($CV_{BS}$).

5. **Layer 4 (Structural & Gradient Diagnostics)**:
   - Shannon entropy $H(P) = -\sum p \log p$ and sparsity percentage of attention maps.
   - Mixture-of-Experts (MoE) router load balance coefficient of variation.
   - Vanishing / exploding gradient norm auditor.

6. **Comprehensive Automated Reporting**:
   - Incremental JSON persistence (`results.json`).
   - Summary table (`summary.csv`).
   - Markdown leaderboard with composite scores (`leaderboard.md`).
   - Training loss curves, scaling curves, and multi-metric radar charts (`plots/`).

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r benchmark/requirements.txt
```

### 2. Fast Smoke Test (Toy verification of 2–3 models)
```bash
python3 benchmark/run_benchmark.py --smoke-test --models full,svd_linear
```

### 3. Full Benchmark Run
```bash
python3 benchmark/run_benchmark.py --config benchmark/configs/default.yaml
```

### 4. Custom Task / Model Subset
```bash
python3 benchmark/run_benchmark.py --models full,svd_linear,performer,mamba --tasks sst2,mrpc
```

---

## 📊 Composite Scoring Formula

The benchmark computes a composite performance score ($0 \text{ to } 100$):

$$\text{Score} = 0.40 \cdot \text{Quality} + 0.20 \cdot \text{Efficiency} + 0.20 \cdot \text{Stability} + 0.20 \cdot \text{Robustness}$$

Where:
- **$\text{Quality}$**: $\min\left(120, \frac{\text{Avg GLUE}_{\text{model}}}{\text{Avg GLUE}_{\text{full}}} \times 100\right)$
- **$\text{Efficiency}$**: $50 \times \left(0.6 \cdot \min(\text{SpeedRatio}, 2.0) + 0.4 \cdot \min(\text{MemoryRatio}, 2.0)\right)$
- **$\text{Stability}$**: $\frac{\text{Successful Seeds}}{\text{Total Seeds}} \times 100$
- **$\text{Robustness}$**: $\max(0, (1 - CV_{LR}) \times 100)$

---

## 📁 Output Artifacts

```text
benchmark_output/
├── results.json                   # Machine-readable detailed benchmark output
├── summary.csv                    # Filterable tabular dataset
├── leaderboard.md                 # Publication-ready ranking table
├── plots/
│   ├── training_curves/           # Step-wise loss curves per model & task
│   ├── scaling/                   # Throughput scaling vs sequence length
│   └── radar_charts/              # Multi-axial architecture comparison radar
└── logs/
    ├── benchmark.log              # Full execution stream
    └── errors.log                 # Error and warning tracebacks
```

# Experiment Results Ledger & Provenance API

NeuralOps features an immutable, dual-storage Experiment Results Ledger designed for scientific reproducibility, cryptographic provenance, and zero-fake verification.

Every benchmark execution, kernel profile, and downstream fine-tuning run is recorded with exact hardware telemetry, Git commit SHAs, driver versions, and physical timestamp metadata.

[![Release](https://img.shields.io/badge/release-v2.0.6-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.0.6)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. Dual-Storage Architecture

The results ledger synchronizes across two complementary storage formats:

1. **Human-Readable JSONL (`.jsonl`)**: Git-friendly line-delimited JSON enabling fine-grained diff tracking, commit review, and stream parsing.
2. **High-Performance SQLite (`.sqlite`)**: Indexed relational database supporting rapid SQL aggregation, multi-dimensional queries, and automated LaTeX table generation.

```
+-------------------------------------------------------------------+
|               NeuralOps Experiment Results Ledger                 |
+-------------------------------------------------------------------+
           |                                             |
           v                                             v
  Git-Friendly Stream                            Indexed Query Engine
  results_ledger.jsonl                           results_ledger.sqlite
  (Full Run Telemetry)                           (Relational Projections)
           \                                             /
            +--------------------+----------------------+
                                 |
                                 v
        Hugging Face Hub Master Benchmark Dataset:
        https://huggingface.co/datasets/Hooshaai/neuralops-results
```

---

## 2. Official Hugging Face Dataset

All benchmark records, raw telemetry logs, and SQLite database snapshots are mirrored and versioned in the official Hugging Face benchmark repository:

👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

### Repository Contents
- `results_master.sqlite`: SQLite master ledger containing all 203 verified module runs, 131 attentions, and 72 compression sweeps.
- `benchmarks_telemetry.jsonl`: Timestamped hardware telemetry logs with GPU temperature, VRAM allocation, and arithmetic intensity.
- `latex_tables/`: Publication-grade `.tex` tables matching verified ledger figures for empirical papers.

---

## 3. Physical Hardware Telemetry Invariants

All empirical data points in the results ledger are grounded in physical execution on:
- **Silicon**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102 silicon)**.
- **Hardware Profile**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Institution**: Department of Computer Engineering, College of Engineering, University of Tehran.

---

## 4. Python API Usage

### Recording an Experiment Result

```python
import neuralops
from neuralops.analysis.results_ledger import ResultsLedger, ResultRecord

# Initialize ledger pointing to local repository results
ledger = ResultsLedger()

# Record benchmark execution with provenance
record = ResultRecord(
    record_id="rec_gdn_scan_001",
    module_name="gdn_scan",
    category="attention",
    family="delta_recurrence",
    version_tag="v2.1.0",
    status="PASS",
    timestamp_utc="2026-11-06T12:00:00Z",
    git_commit_sha="a1b2c3d4",
    generation_method="real_execution",
    experimental_setup={"sequence_length": 2048, "batch_size": 2, "dtype": "float16"},
    metrics={"latency_ms": 0.364, "speedup": 4.51, "peak_vram_mb": 264.0},
    notes="Ampere RTX 3090 SM 8.6 verified execution",
)
ledger.add_or_replace(record)
```

### Querying the Ledger

```python
# Query verified records filtered by parameters
records = ledger.query(
    category="attention",
    max_latency_ms=1.0,
    sort_by="timestamp_utc",
    descending=True,
    limit=10,
)

for rec in records:
    latency = rec.metrics.get("latency_ms", 0.0)
    print(f"{rec.module_name:<25} | Latency: {latency:.3f} ms | Status: {rec.status}")
```

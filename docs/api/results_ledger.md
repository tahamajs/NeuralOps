# Experiment Results Ledger & Provenance API

NeuralOps features an immutable, dual-storage Experiment Results Ledger designed for scientific reproducibility, cryptographic provenance, and zero-fake verification.

Every benchmark execution, kernel profile, and downstream fine-tuning run is recorded with exact hardware telemetry, Git commit SHAs, driver versions, and physical timestamp metadata.

[![Release](https://img.shields.io/badge/release-v1.6.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v1.6.0)
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
- **Silicon**: **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon)**.
- **Hardware Profile**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Institution**: Department of Computer Engineering, College of Engineering, University of Tehran.

---

## 4. Python API Usage

### Recording an Experiment Result

```python
import neuralops
from neuralops.benchmarks.results_ledger import ResultsLedger, BenchmarkRecord

# Initialize ledger pointing to local storage
ledger = ResultsLedger(
    jsonl_path="results/results_ledger.jsonl",
    sqlite_path="results/results_ledger.sqlite",
)

# Record benchmark execution with provenance
record = BenchmarkRecord(
    module_name="gdn_scan",
    family="delta_recurrence",
    model="gpt2",
    sequence_length=2048,
    batch_size=2,
    dtype="float16",
    latency_ms=0.364,
    speedup=4.51,
    peak_vram_mb=264.0,
    arithmetic_intensity=138.5,
    achieved_tflops=36.4,
    status="PASS_FAST",
    hardware="NVIDIA GeForce RTX 4070 (SM 8.9)",
)
ledger.record(record)
```

### Querying the Ledger

```python
# Query top-performing kernels by speedup
top_kernels = ledger.query_sql("""
    SELECT module_name, speedup, achieved_tflops, peak_vram_mb
    FROM benchmarks
    WHERE status = 'PASS_FAST' AND sequence_length = 2048
    ORDER BY speedup DESC
    LIMIT 10
""")

for row in top_kernels:
    print(f"{row['module_name']:<25} | {row['speedup']:.2f}x | {row['achieved_tflops']:.1f} TFLOPS")
```

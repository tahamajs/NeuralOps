# 💻 NeuralOps CLI Reference Guide (v2.1.0)

The canonical entry points for the NeuralOps command line interface are:
- `neuralops <subcommand>`
- `nops <subcommand>`
- `python -m neuralops.benchmarks <subcommand>`

The CLI is implemented via `neuralops.benchmarks.cli` using `argparse` with structured console diagnostics and optional Rich formatting.

---

## 1. Global Flags & Architecture Policy

In accordance with [AGENTS.md Section 10](file:///Users/tahamajs/dev/Project/AGENTS.md#L273), the CLI follows strict configuration and operational contracts:
- **Configuration Precedence:** `defaults < config.yaml < environment variables < explicit CLI flags`.
- **Fail-Closed Execution:** Explicit `--device cuda` fails closed if CUDA is unavailable; execution never silently falls back to CPU or mocks execution.
- **Output Streaming:** When `--json` is supplied, stdout contains valid JSON only; progress diagnostics and notices are routed to stderr or suppressed.
- **Dry-Run Mode:** `--dry-run` / `--plan` estimates parameter spaces and schedules without allocating heavy GPU tensors or mutating files.

### Global Options
| Option | Type | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | Flag | Display comprehensive usage overview, grouped commands, and quickstart examples |
| `-v`, `--version` | Flag | Print the installed NeuralOps release version |
| `--json` | Flag | Format stdout as structured JSON where applicable |
| `-q`, `--quiet` | Flag | Suppress non-essential banners, headers, and interactive progress bars |
| `--verbose` | Flag | Print granular execution traces, backend attempts, and full stack traces |
| `--install-completion {bash,zsh,fish}` | String | Generate shell completion scripts for your environment |

### Exit Codes Specification (AGENTS.md §10)
| Exit Code | Symbol | Description |
| :---: | :--- | :--- |
| `0` | `SUCCESS` | Command completed successfully within all specified tolerances |
| `1` | `BENCHMARK_ERROR` | Runtime failure, execution error, or benchmark abort |
| `2` | `USAGE_CONFIG_ERROR` | Unrecognized argument, missing required flag, or invalid parameter |
| `3` | `BACKEND_UNAVAILABLE` | Requested hardware accelerator (e.g., CUDA, Triton) unavailable |
| `4` | `FAIL_NUMERICS` | Numerical tolerance exceeded (`max_diff > tolerance`) or regression gate failed |
| `5` | `PARTIAL_RUN` | Sweep stopped prematurely or partial results returned |
| `6` | `OUTPUT_SCHEMA_ERROR` | Persistence validation failed or output JSON/schema mismatch |
| `70` | `INTERNAL_SOFTWARE_ERROR`| Unexpected internal software defect or unhandled fatal condition |
| `130` | `SIGINT` | Process interrupted by user (`Ctrl+C`) |

---

## 2. Command Hierarchy Overview

```text
neuralops
  ├── benchmark
  │     ├── run        # Execute individual or batch micro-benchmark suites
  │     ├── plan       # Dry-run execution grid estimation without GPU allocation
  │     ├── list       # Display registered attention & compressor catalog
  │     ├── info       # Display detailed technical specs for a given module
  │     ├── compare    # Regression test comparison between baseline and candidate
  │     ├── resume     # Safely resume an interrupted benchmark sweep
  │     ├── matrix     # Multi-axis sequence length, batch size, and dimension grid
  │     └── multiseed  # Multi-seed statistical significance & cold-cache testing
  ├── matrix           # Top-level alias for multi-dimensional grid sweeps
  ├── list             # List and filter registered attention modules & compressors
  ├── info             # Detailed GPU hardware telemetry and registry metrics
  ├── verify           # Fast environment sanity check (PyTorch, CUDA, MPS, counts)
  ├── doctor           # System diagnostic, toolchain, and invariant validator
  ├── transplant       # In-place attention surgery on existing Hugging Face models (alias: surgery)
  ├── compress         # Model surgery compression via 82 compression algorithms
  ├── search           # AutoSVD Pareto optimal rank allocation via Fisher sensitivity
  ├── export           # Export compressed/surgerized models to safetensors, ONNX, or TorchScript
  ├── serve            # High-throughput vLLM / SGLang compatible serving server
  └── inspect          # Technical architecture specification sheet for registered modules
```

---

## 3. Core Subcommands

### 1. `neuralops benchmark run` (alias: `neuralops run`)
Executes micro-benchmarks across attention mechanisms, compressors, and CUDA/Triton kernels with autograd gradient checking and cold-cache eviction.

```bash
# Basic run on NVIDIA CUDA hardware with FP16 precision
neuralops benchmark run --benchmark attention:flash_attention --device cuda --dtype fp16

# Run full attention evaluation on fine-tuned Hooshaai models
neuralops benchmark run \
    --model Hooshaai/bert-flash_attention-sst2 \
    --dataset sst2 \
    --device cuda \
    --batch-size 32 \
    --seq-len 128 \
    --dtype fp16 \
    --output-dir results/bert_eval/

# Benchmark low-rank compression on Hooshaai DistilBERT
neuralops benchmark run \
    --benchmark compressor:hnc_svd \
    --model Hooshaai/distilbert-base-fwsvd-rank64 \
    --rank 32 \
    --device cuda \
    --output-dir results/compression_sweep/

# Run cold-cache benchmarking with explicit L2 eviction
neuralops benchmark run \
    --module gated_delta_net \
    --device cuda \
    --warmup 25 \
    --batch-size 4 \
    --seq-len 2048 \
    --output-dir results/microbench/

# Fail-closed behavior on unavailable hardware (exits with code 1 / 3)
neuralops benchmark run --benchmark custom_cuda_kernel --device cuda
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--benchmark` | String | None | Canonical benchmark key (e.g. `attention:cosformer`, `cuda:act_scale_gemm`) |
| `--module` | String | None | Single module name (alias for `--benchmark`) |
| `--modules` | String | None | Comma-separated operation keys |
| `--category` | Choice | `all` | Category: `attention`, `compressor`, `cuda`, `triton`, `roofline`, `quantization`, `moe`, `serving`, `rope`, `sampling`, `ssm`, `varlen`, `sweep`, `all` |
| `--model` | String | `distilbert-base-uncased` | Base model identifier (e.g., `Hooshaai/bert-flash_attention-sst2`) |
| `--models` | String | None | Comma-separated list of model identifiers |
| `--dataset` | String | `sst2` | Target dataset for evaluation |
| `--rank` | Integer | `64` | Target rank for low-rank attention or compressor modules |
| `--device` | String | `cuda` | Hardware target: `cuda`, `mps`, `cpu` (fails closed if CUDA unavailable) |
| `--dtype` | Choice | `fp32` | Precision format: `fp16`, `bf16`, `fp32` |
| `--batch-size` | Integer | `32` | Evaluation batch size |
| `--seq-len` | Integer | `128` | Evaluation sequence length |
| `--output-dir` | String | `results` | Directory where JSON, JSONL, and summary reports are persisted |
| `--dry-run` | Flag | False | Print plan without allocating tensors or executing runs |
| `--all-attn` | Flag | False | Run every registered attention mechanism |
| `--all-comp` | Flag | False | Run every registered compression engine |

---

### 2. `neuralops matrix` (or `neuralops benchmark matrix`)
Orchestrates multi-axis parameter grid sweeps across sequence lengths, batch sizes, hidden dimensions, dtypes, and causal masking modes.

```bash
# Multi-axis sequence length and batch saturation sweep
neuralops matrix \
    --modules attention:flash_attention,attention:gla,attention:gated_delta_net \
    --seq-lens 128,512,1024,2048 \
    --batch-sizes 1,2,4,8 \
    --hidden-dims 64,128,256 \
    --dtypes fp16,bf16 \
    --causal \
    --warmup 10 \
    --iters 50 \
    --device cuda \
    --output-dir results/matrix_sweep/

# Dry-run execution grid plan without allocating memory
neuralops matrix \
    --modules attention:cosformer \
    --seq-lens 256,1024,4096 \
    --batch-sizes 1,4 \
    --plan
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--modules`, `--module` | String | `attention:full_attention` | Comma-separated module names |
| `--seq-lens` | String | `128,512,1024,2048` | Comma-separated sequence lengths |
| `--batch-sizes` | String | `1,2,4,8` | Comma-separated batch sizes |
| `--hidden-dims` | String | `64,128,256,512` | Comma-separated hidden dimensions |
| `--dtypes` | String | `float32` | Comma-separated dtypes (`fp32`, `fp16`, `bf16`) |
| `--causal` | Flag | None | Enforce causal masking for all matrix grid points |
| `--no-causal` | Flag | None | Enforce bidirectional (non-causal) attention |
| `--causal-modes` | String | None | Comma-separated boolean modes (e.g. `true,false`) |
| `--device` | String | `cpu` | Execution device (`cuda`, `mps`, `cpu`) |
| `--warmup` | Integer | `5` | Warmup iterations per cell |
| `--iters` | Integer | `20` | Timed iterations per cell |
| `--flush-l2` | Flag | True | Evict GPU L2 cache before each timing cycle |
| `--no-flush-l2` | Flag | False | Disable L2 cache eviction |
| `--plan`, `--dry-run` | Flag | False | Print planned parameter combinations without execution |
| `--json` | Flag | False | Output execution matrix summary in JSON format |

---

### 3. `neuralops list` (or `neuralops benchmark list`)
Queries and displays the canonical registry of attention mechanisms, compression engines, and accelerated GPU kernels.

```bash
# List all registered attention mechanisms
neuralops list --type attention

# Filter for Triton-accelerated linear attention mechanisms
neuralops list --type attention --family "Linear" --backend triton

# Filter for CUDA-accelerated compression algorithms
neuralops list --type compressor --backend cuda

# Export complete catalog as structured JSON
neuralops list --json > catalog_export.json
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--type` | Choice | `all` | Registry category: `attention`, `compressor`, `kernels`, or `all` |
| `--family` | String | None | Filter mechanisms by mathematical family substring (e.g., `SSM`, `Linear`, `SVD`) |
| `--backend` | Choice | `all` | Filter by accelerator backend support: `cuda`, `triton`, `pytorch`, `all` |
| `--supports` | Choice | None | Filter by architectural capability: `causal`, `gqa`, `rope`, `cuda` |
| `--json` | Flag | False | Output full catalog records formatted as JSON |

---

### 4. `neuralops info` (or `neuralops benchmark info`)
Displays deep technical telemetry for local GPU hardware or technical specifications for a specific registered module.

```bash
# Print local host, GPU hardware, and driver telemetry
neuralops info

# Inspect architectural specification for a specific attention mechanism
neuralops info attention:gated_delta_net

# Inspect technical specification for a compression engine
neuralops info compressor:hnc_svd

# Export info payload as JSON
neuralops info --json
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String (Positional) | None | Optional module or benchmark name to inspect (e.g., `attention:cosformer`) |
| `--json` | Flag | False | Output technical specifications as structured JSON |

---

### 5. `neuralops verify`
Executes an instant zero-overhead sanity check on system hardware, PyTorch runtimes, CUDA/MPS acceleration, and physical repository registry counts.

```bash
# Fast terminal verification check
neuralops verify

# Machine-readable JSON output for CI pipelines
neuralops verify --json
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--json` | Flag | False | Format output as structured JSON |

#### Example Output (`--json`)
```json
{
  "platform": "macOS-27.2-arm64-arm-64bit-Mach-O",
  "python_version": "3.14.3",
  "pytorch_version": "2.11.0",
  "cuda_available": false,
  "device_name": "N/A",
  "device_count": 0,
  "mps_available": true,
  "triton_available": false,
  "attention_registry_count": 115,
  "compressor_registry_count": 82,
  "triton_kernels_count": 62,
  "cuda_kernels_count": 148,
  "status": "PASS"
}
```

---

### 6. `neuralops doctor`
Performs exhaustive system health diagnostics, toolchain discovery, and repository invariant certification.
- **Hardware & Drivers:** CUDA runtime, NVIDIA Driver version, CUDA Compute Capability (SM), and GPU VRAM capacity.
- **Compiler Toolchains:** Verifies availability of `nvcc`, GCC / Host C++17 compiler, and Clang.
- **Triton JIT Environment:** Verifies Triton compilation headers, cache path, and PTX generation capability.
- **Subpackage Integrity:** Verifies clean imports across all core subpackages (`neuralops.core`, `neuralops.ops`, `neuralops.layers`, `neuralops.benchmarks`, `neuralops.export`).

```bash
# Run comprehensive health diagnostic
neuralops doctor

# Export diagnostic ledger as JSON
neuralops doctor --json
```

#### Key Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--json` | Flag | False | Output diagnostic results in JSON format |

- Exits with return code `0` if all essential toolchains and diagnostics pass.
- Exits with return code `1` if any prerequisite compiler or critical dependency fails.

---

### 7. `neuralops benchmark compare` (alias: `neuralops compare`)
Enforces automated numerical regression gates between candidate runs and verified baselines.

```bash
neuralops benchmark compare \
    --baseline results/baselines/candidate_baseline.json \
    --candidate results/run_01/summary.json \
    --metric tokens_per_sec \
    --max-regression-pct 5.0
```
- Exits with return code `0` if candidate performs within tolerance.
- Exits with return code `4` (`FAIL_NUMERICS`) if performance regression $> 5.0\%$.

---

### 8. `neuralops transplant` (alias: `neuralops surgery`)
Performs 1-line surgical attention transplantation and Ridge closed-form subspace calibration on pretrained Hugging Face models.

```bash
neuralops transplant \
    --model Hooshaai/bert-flash_attention-sst2 \
    --attention gated_delta_net \
    --rank 64 \
    --calibrate \
    --calibration-text "Sample calibration text for subspace alignment"
```

---

## 4. End-to-End Workflow Examples with Hooshaai Models

### Workflow A: Benchmark Fine-Tuned Model with Low-Rank Attention
```bash
# 1. Verify runtime environment
neuralops verify

# 2. Benchmark fine-tuned Hooshaai BERT with Flash Attention
neuralops benchmark run \
    --model Hooshaai/bert-flash_attention-sst2 \
    --dataset sst2 \
    --device cuda \
    --dtype fp16 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir results/hoosha_bench/

# 3. Check performance regression against certified baseline
neuralops benchmark compare \
    --baseline results/baselines/bert_sst2_baseline.json \
    --candidate results/hoosha_bench/summary.json \
    --metric tokens_per_sec \
    --max-regression-pct 2.0
```

### Workflow B: Multi-Parameter Sequence Saturation Sweep
```bash
neuralops matrix \
    --modules attention:flash_attention,attention:gated_delta_net \
    --seq-lens 128,512,1024,2048 \
    --batch-sizes 1,4,16 \
    --dtypes fp16 \
    --device cuda \
    --flush-l2 \
    --output-dir results/saturation_sweep/
```

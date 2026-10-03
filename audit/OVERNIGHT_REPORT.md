# Overnight Autonomous Run Status Report

**Session ID**: `25751b5b-65bd-4114-86a3-902417a3a6a9`  
**Timestamp**: 2026-09-25T02:22:00+03:30 (Local) / 2026-09-24T22:52:00Z (UTC)  
**Elapsed Autonomous Time**: ~2.5 Hours  
**Target File**: `audit/OVERNIGHT_REPORT.md`  

---

## 1. Executive Summary & GPU Telemetry

The overnight autonomous test matrix is actively running on the GPU server. All jobs were launched using persistent shell background processes (`nohup ... &`) with dynamic VRAM contention protection.

* **GPU Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU (AD106, 8,188 MiB GDDR6)
* **GPU Utilization**: **100% Compute**
* **Power Draw**: **83.0 W** (P0 State)
* **Temperature**: **72°C** (well within thermal ceiling of 80°C)
* **Memory Allocated**: **4,436 MiB / 8,188 MiB** (healthy margin avoiding OOM)
* **Active Background Jobs**: 5 parallel jobs active.

---

## 2. Completed Job Status Matrix

### Completed Runs Summary (21 Configurations Passing)

#### DistilBERT (SST-2 Benchmark)
| Module | Rank | Accuracy (%) | F1 Score | Status | Run Time (s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `a3_compressor` | 16 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `a3_compressor` | 32 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `a3_compressor` | 64 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `a3_compressor` | 128 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `a3_compressor` | 256 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `a3_compressor` | 384 | Complete | Complete | **PASS** | ~30s | Complete sweep |
| `adasvd_compressor` | 256 | Complete | Complete | **PASS** | ~45s | Verified |
| `adasvd_compressor` | 384 | Complete | Complete | **PASS** | ~45s | Verified |
| `svd_llm_compressor` | 16 | Complete | Complete | **PASS** | ~50s | Verified |
| `svd_llm_compressor` | 32 | Complete | Complete | **PASS** | ~50s | Verified |
| `svd_llm_compressor` | 64 | Complete | Complete | **PASS** | ~50s | Verified |
| `svd_llm_compressor` | 128 | Complete | Complete | **PASS** | ~50s | Verified |

#### RoBERTa (SST-2 Benchmark)
| Module | Rank | Accuracy (%) | F1 Score | Compression | FLOPs Red. | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `adasvd_compressor` | 16 | **81.08%** | 0.8352 | 1.1892x | 70.05% | **PASS** |
| `adasvd_compressor` | 32 | **87.73%** | 0.8766 | 1.1892x | 70.05% | **PASS** |
| `adasvd_compressor` | 64 | **87.73%** | 0.8766 | 1.1892x | 70.05% | **PASS** |
| `a3_compressor` | 256 | Complete | Complete | 1.25x | Complete | **PASS** |
| `a3_compressor` | 384 | Complete | Complete | 1.15x | Complete | **PASS** |

#### GPT-2 (WikiText-2 Perplexity Benchmark)
| Module | Rank | Status | Notes |
| :--- | :--- | :--- | :--- |
| `a3_compressor` | 256 | **PASS** | Completed full sequence eval |
| `svd_llm_compressor` | 16 | **PASS** | Truncation-aware SVD |
| `svd_llm_compressor` | 32 | **PASS** | Truncation-aware SVD |
| `svd_llm_compressor` | 64 | **PASS** | Truncation-aware SVD |

#### Job 1: SSM Sequence Length Scaling (`models/gpt2`, $B=16$)
* $N=128$: Forward: **2,158.06 ms** (949.0 tok/s), Peak VRAM: **2,448.14 MB** [**PASS**]
* $N=256$: Forward: **7,367.31 ms** (556.0 tok/s), Peak VRAM: **2,907.79 MB** [**PASS**]
* $N=512$: In progress (PID 28640 actively computing).

---

## 3. Auto-Recovery & Engineering Fixes Applied

1. **Subprocess `PYTHONPATH` Resolution in Isolated Worker**:
   * *Problem*: In `neuralops/run_one_module_isolated.py`, subprocesses spawned without `Codes` explicitly prepended to `sys.path` failed with `ModuleNotFoundError: No module named 'framework'`.
   * *Fix*: Applied path bootstrapping directly at the head of `run_one_module_isolated.py` before module imports (`sys.path.insert(0, str(ROOT / 'Codes'))`). Deployed and verified on GPU.
2. **Crash Dummy Cleanup**:
   * 12 empty `summary.csv` placeholders generated during the initial 0.1s import failure were purged from the sweep directories so that the respective rank configurations auto-re-evaluate cleanly.
3. **Dynamic VRAM Backoff**:
   * Shell loops in `scripts/run_job*.sh` throttle launches when GPU memory exceeds 4,500 MiB, preventing CUDA out-of-memory cascades on the 8GB GPU.

---

## 4. Current Pipeline ETA

* **Job 1 (SSM Scaling)**: `mamba` at $N=512, 1024$ completing shortly, followed by remaining SSM modules. ETA: ~2-3 hours.
* **Job 2 (Linformer Scaling)**: Queued behind Job 1 VRAM window. ETA: ~15 minutes.
* **Jobs 3, 4, 5 (Rank Sweeps)**: Progressing continuously across ranks 16, 32, 64, 128, 256, 384. Full matrix completion expected by morning (~06:00 - 07:00 local time).

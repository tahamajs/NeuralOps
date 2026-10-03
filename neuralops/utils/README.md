# Utilities Subsystem (`Codes/framework/utils`)

The `utils` subsystem provides essential shared helper utilities, device management, seed reproducibility, checkpoint managers, execution profiling, layer normalization precision patches, and universal module wrappers.

---

## 🏗 Architecture Overview

- **Device Management (`device.py`)**: Automatic hardware probe for NVIDIA CUDA, Apple Silicon MPS, and CPU devices with memory cache management.
- **Reproducibility Engine (`seed.py`)**: Deterministic seed setting across Python `random`, NumPy, PyTorch CPU, and CUDA backends.
- **Universal Attention Wrapper (`replace_attention.py`)**: Standardized forward adapter wrapping custom attention modules to guarantee signature compatibility with standard transformer blocks.
- **Checkpoint Manager (`checkpoint.py`)**: Atomic checkpoint serialization, top-k best model retention, and resume-from-checkpoint utilities.
- **Execution Profiler (`profiler.py`)**: Micro-benchmark timers, CUDA event-based kernel latency profilers, and peak VRAM allocators.
- **Numerical Stability Patches (`layer_norm_fix.py`)**: FP32 LayerNorm/RMSNorm precision stabilization preventing underflow during mixed-precision training.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `device.py` | `get_device()`, multi-GPU device selection, and memory cache cleanup. |
| `seed.py` | `set_seed()` for deterministic, reproducible training runs. |
| `replace_attention.py` | `UniversalAttentionWrapper` ensuring uniform attention forward calling conventions. |
| `checkpoint.py` | Safe atomic model checkpoint saving and loading utilities. |
| `profiler.py` | Context-manager and decorator latency/VRAM profiling tools. |
| `layer_norm_fix.py` | Precision upcasting hooks for LayerNorm during FP16 training. |
| `paths.py` | Standardized filesystem path resolvers for checkpoints, logs, and datasets. |

---

## 🚀 Quick Start & Usage

### 1. Setting Global Seed & Device
```python
from framework.utils.device import get_device
from framework.utils.seed import set_seed

set_seed(42)
device = get_device()  # Returns torch.device('cuda', 'mps', or 'cpu')
```

### 2. Profiling Execution Latency
```python
from framework.utils.profiler import Timer

with Timer("Model Forward Pass"):
    outputs = model(input_ids)
```

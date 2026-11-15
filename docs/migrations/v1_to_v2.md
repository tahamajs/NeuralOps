# Migration Guide: NeuralOps v1.x → v2.x

This document provides architectural and practical upgrade guidance for enterprise deployments, researchers, and engineers transitioning codebases from NeuralOps v1.x monolithic namespaces to the modular v2.x architecture.

---

## Architecture Paradigm Shift

NeuralOps v1.x utilized flat, coupled namespaces where layer definitions, compression implementations, and training logic were tightly interleaved. NeuralOps v2.x establishes strict architectural boundaries:
- **Clean Subpackage Separation:** Attention mechanisms and layers live in `neuralops.layers`, compression engines in `neuralops.compression`, training and distillation regimes in `neuralops.trainer`, and model surgery operations in `neuralops.surgery`.
- **Lazy Canonical Registry:** Registry discovery is strictly lazy without import-time CUDA or filesystem side effects.
- **Enterprise Security Hardening:** Default hub namespace migrated to `Hooshaai/`, servers bind strictly to `127.0.0.1`, and `trust_remote_code` defaults strictly to `False`.
- **Unified CLI Entrypoint:** All commands (`benchmark`, `surgery`, `matrix`, `serve`, `tui`, `ui`) route through the canonical `neuralops` binary.

---

## Complete Namespace Mapping

| v1.x Legacy Path | v2.x Canonical Path | Description & Notes |
| :--- | :--- | :--- |
| `neuralops.attention.*` | `neuralops.layers.attention.*` | Standard attention mechanisms (Cosformer, SVDLinear, Flash) |
| `neuralops.attentions.*` | `neuralops.layers.*` | Layer definitions and modular attention blocks |
| `neuralops.compress.*` | `neuralops.compression.*` | Singular value decomposition, quantization, and pruning engines |
| `neuralops.compression_engine.*` | `neuralops.compression.*` | Deprecated engine alias forwarding to `neuralops.compression` |
| `neuralops.core.compression_engine.*`| `neuralops.compression.*` | Core compression primitives |
| `neuralops.train.*` | `neuralops.trainer.*` | Modernized training loops, 6-pillar KD, and online TTT |
| `neuralops.training.*` | `neuralops.trainer.*` | Deprecated training module forwarding to `neuralops.trainer` |
| `neuralops.surgery.*` | `neuralops.surgery.*` | Universal surgery director and delta serialization |
| `neuralops.surgery.universal_calibrator` | `neuralops.trainer.regimes` | Distillation and post-surgery calibration regimes |
| `neuralops.core.rope` | `neuralops.core.utils.rope` | Rotary position embedding primitives |
| `neuralops.core.gqa_utils` | `neuralops.utils.architecture` | Grouped-Query Attention utilities |

---

## Example 1: Upgrading a Fine-Tuning & Distillation Script

### Before (v1.x)
In v1.x, attention mechanisms, compressors, and training routines were imported from flat top-level packages:
```python
import torch
from neuralops.attention import CosformerAttention
from neuralops.compress import SVDCompressor
from neuralops.train import DistillationTrainer

# Instantiate linear attention module
attn = CosformerAttention(dim=768, num_heads=12)

# Compress model weights
compressor = SVDCompressor(rank=64)

# Initialize trainer
trainer = DistillationTrainer(
    student_model=torch.nn.Identity(),
    teacher_model=torch.nn.Identity(),
    learning_rate=1e-4,
)
```

### After (v2.x)
In v2.x, imports are strictly organized under `layers`, `compression`, and `trainer`:
```python
import torch
from neuralops.layers.attention import CosformerAttention
from neuralops.compression import SVDCompressor
from neuralops.trainer import DistillationTrainer

# Instantiate linear attention module with identical constructor interface
attn = CosformerAttention(dim=768, num_heads=12)

# Compress model weights
compressor = SVDCompressor(rank=64)

# Distillation trainer with enhanced telemetry and multi-loss supervision
trainer = DistillationTrainer(
    student_model=torch.nn.Identity(),
    teacher_model=torch.nn.Identity(),
    learning_rate=1e-4,
)
```

---

## Example 2: Canonical Registry Access & Aliases

### Before (v1.x)
In v1.x, duplicate or ambiguous string aliases were permitted:
```python
# Ambiguous aliases in v1.x
from neuralops.benchmarks.registry import BenchmarkRegistry

# Multiple string keys pointed to disparate variants
cls1 = BenchmarkRegistry.get("svd_linear_attn")
cls2 = BenchmarkRegistry.get("svd_linear_inject")
```

### After (v2.x)
In v2.x, every unique operator has exactly one canonical key, with transparent aliasing registered centrally:
```python
from neuralops.benchmarks.registry import get_registry

# Lazy registry without import-time side effects
registry = get_registry()

# Canonical identifier
cls = registry.get("svd_linear")
```

---

## Example 3: CLI Invocation — Web UI vs Terminal UI

In v1.x, `neuralops ui` was overloaded to handle both terminal and graphical interfaces. In v2.x, the commands are explicitly decoupled:

### Before (v1.x)
```bash
# Terminal snapshot in v1.x
neuralops ui --snapshot
```

### After (v2.x)
```bash
# Modern Web UI (FastAPI + htmx dashboard)
neuralops ui --host 127.0.0.1 --port 8080

# Dedicated Terminal UI (interactive 2-pane curses inspector)
neuralops tui --snapshot
```

---

## Example 4: Hugging Face Hub Namespace & Security Defaults

In v1.x, model checkpoints and zoo weights were loaded under personal user namespaces. In v2.x:
1. The official default organization is `Hooshaai`.
2. Safe deserialization is enforced (`weights_only=True`).
3. External code execution (`trust_remote_code`) defaults strictly to `False`.

### Before (v1.x)
```python
from neuralops.hub import load_model

# Unsafe remote code execution allowed by default
model = load_model("personal-user/qwen-1.5b-cosformer")
```

### After (v2.x)
```python
from neuralops.hub import load_model

# Loads from official Hooshaai organization with hardened security defaults
model = load_model(
    "Hooshaai/qwen-1.5b-cosformer",
    trust_remote_code=False,
)
```

---

## Example 5: PyTorch Fallback Reference Implementations

PyTorch fallback reference implementations (e.g. for numerical parity checks and GPU vs CPU testing) have been centralized in `neuralops.ops.pytorch`.

### Before (v1.x)
```python
# Internal unexported test helper imports in v1.x
from neuralops.tests.helpers.ops import gdn_scan_pytorch
```

### After (v2.x)
```python
# Canonical public export in v2.x
from neuralops.ops.pytorch import gdn_scan_pytorch, fused_rmsnorm_pytorch
```

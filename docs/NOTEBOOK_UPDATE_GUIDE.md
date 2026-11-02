# NeuralOps Notebook Standard & Update Guide

This document defines the architectural and coding standards for all 26 Jupyter demonstration notebooks in the [`notebooks/`](../notebooks/) directory of the NeuralOps repository.

---

## 1. Core Operating Principles for Notebooks

All notebooks must follow these standards to ensure reproducibility across developer workstations (macOS CPU/MPS), CI environments, and remote GPU accelerator clusters:

1. **Zero Silent Failures**: Notebooks must never execute an operation that fails silently or produces fake data. If an optional dependency or hardware requirement is missing, the cell must explicitly report `[SKIPPED]` with a clear rationale.
2. **Offline Self-Sufficiency**: Example data in notebooks must be generated synthetically or loaded from local assets. Never require external Hugging Face dataset downloads or network credentials for standard execution.
3. **Execution Determinism**: Every notebook that uses stochastic initialization or synthetic inputs must set explicit seeds (`torch.manual_seed(42)`).

---

## 2. Mandatory Structural Cells

Every notebook in `notebooks/` must begin with the following two standardized cells:

### Cell 1: Prerequisites & Hardware Classification (Markdown)
The very first cell of every notebook must be a Markdown cell titled `# Prerequisites & Hardware Requirements`:

```markdown
# Prerequisites & Hardware Requirements

- **Compute Target**: CPU-compatible (Standard execution) / GPU-accelerated (Triton & CUDA Kernels)
- **Minimum VRAM**: 4 GB (when running on CUDA)
- **External Dependencies**: `torch >= 2.1`, `neuralops >= 2.0.1`, `transformers`
- **Offline Safe**: Yes — utilizes offline synthetic token generators.
```

### Cell 2: Standardized Environment Check (Code)
The second cell must be an executable code cell that inspects the runtime environment and prints version telemetry:

```python
import sys
import torch
import neuralops

print("=" * 60)
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"NeuralOps Version : {neuralops.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Accelerator   : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability   : {torch.cuda.get_device_capability(0)}")
print("=" * 60)
```

---

## 3. Rules for Adding New Sections to Notebooks

When adding new demonstration sections, methods, or empirical analyses to existing notebooks:

### 3.1 Preceding Markdown Headings
Every new section must start with a descriptive Markdown heading detailing the mathematical operator or architectural transformation being demonstrated:
```markdown
## Section X: [Subsystem Name] — [Mathematical Formulation / Goal]
Brief 2-3 sentence overview of the operation, input/output tensors, and expected invariant.
```

### 3.2 Import Discipline
All imports must reside at the very top of the code cell. Never place `import` statements inside functions, loops, or mid-cell logic unless explicitly handling an optional integration with `try...except ImportError`.

### 3.3 CPU Safety & Graceful GPU Fallback
If a cell contains CUDA C++ or Triton JIT kernels that cannot execute on CPU:
- Check `torch.cuda.is_available()` before executing.
- Provide an explicit fallback path or print a clear skip notice:
```python
if not torch.cuda.is_available():
    print("[SKIPPED] Native Triton kernel demonstration requires a CUDA GPU.")
else:
    # GPU kernel execution...
```

### 3.4 Concluding Invariants and Evidence Tables
Every demonstration section must conclude with either:
1. An explicit numerical assertion verifying tensor finiteness and parity:
   ```python
   assert torch.isfinite(outputs).all(), "Detected NaN/Inf in model output!"
   print("✓ Output verification passed: finite logits and shape parity confirmed.")
   ```
2. Or a printed Markdown / ASCII result table displaying measured metrics (latency, FLOPs, parameter reduction).

---

## 4. Notebook Checklist

Before submitting updates to any notebook:
- [ ] First cell is the standardized `# Prerequisites & Hardware Requirements` Markdown block.
- [ ] Second cell executes the standardized Environment Check.
- [ ] Cell outputs are cleared of transient warnings or personal system paths.
- [ ] All code cells run top-to-bottom sequentially without error on CPU (with GPU cells cleanly guarded).
- [ ] No hardcoded numbers or fake benchmark values are substituted.

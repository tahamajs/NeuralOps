---
name: roofline-visualizer-generator
description: "Generate publication-grade empirical hardware roofline diagrams and arithmetic intensity plots on SM 8.9"
version: "1.0.0"
tags: ["neuralops", "roofline", "visualization", "hardware", "performance"]
---

# Skill: Hardware Roofline Visualizer and Arithmetic Intensity Modeling

## When to use
Use this skill when analyzing, visualizing, or defending operational hardware bounds across the 131 attention mechanisms and 72 compression engines. It computes empirical arithmetic intensity (FLOP/Byte), models theoretical and achievable ceilings on NVIDIA RTX 4070 (SM 8.9), and exports vector figures for the manuscript.

## Prerequisites
- Manuscript environment: `thesis_final/810101504_main.tex`.
- Remote GPU host: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9).
- Master results ledger: `results/registry/master_ledger.jsonl`.
- Python 3.10+ with `matplotlib` and `numpy`.

## Canonical Audit Files Reference
All roofline parameters and arithmetic intensity calculations must cite the canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced for distillation healing execution FLOPs.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced for low-rank projection arithmetic intensity.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced for the 183.1 FLOP/Byte attention threshold, measured GDDR6 bandwidth ($255.8\text{ GB/s}$), and FP16 compute peak ($70.6\text{ TFLOP/s}$).
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced for memory traffic scaling across sequence lengths.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced for hardware testbed specifications.

## Canonical Repository Invariants
Ensure calculations maintain canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Roofline Modeling and Plotting Methodology

### 1. Hardware Operational Ceilings (SM 8.9)
Model attainable performance using the Williams formulation:
$$\text{Performance (TFLOP/s)} = \min\left(P_{\text{peak}}, I \times B_{\text{peak}}\right)$$
- Peak FP16 Tensor Core Compute: $P_{\text{peak}} = 70.6\text{ TFLOP/s}$.
- Measured GDDR6 Memory Bandwidth: $B_{\text{peak}} = 255.8\text{ GB/s} = 0.2558\text{ TB/s}$.
- Balance Knee Point: $I_{\text{knee}} = P_{\text{peak}} / B_{\text{peak}} = 276.0\text{ FLOP/Byte}$.
- Attention Operational Bound: $I \le 183.1\text{ FLOP/Byte}$ (strictly memory-bound).

### 2. Plotting Script Implementation
Export publication-grade vector PDF and raster PNG:
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_roofline(output_path: str):
    intensities = np.logspace(-1, 3, 500)
    b_peak = 0.2558  # TB/s
    p_peak = 70.6    # TFLOP/s
    
    attainable = np.minimum(p_peak, intensities * b_peak)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(intensities, attainable, color="#1f77b4", linewidth=2, label="Attainable Ceiling (SM 8.9)")
    plt.axvline(x=183.1, color="red", linestyle="--", label="Attention Bound (183.1 FLOP/B)")
    plt.scatter([183.1], [46.8], color="red", zorder=5, label="Standard Softmax Attention")
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
    plt.ylabel("Attainable Performance (TFLOP/s)")
    plt.title("Williams Roofline Model on NVIDIA RTX 4070 (SM 8.9)")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_path}.pdf", format="pdf")
    plt.savefig(f"{output_path}.png", format="png", dpi=300)
    plt.close()
```

---

## Verification
Confirm the roofline modeling satisfies all quality gates:
1. **Invariant Counts**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must confirm 131 / 72 / 26
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Compilation Check**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must compile cleanly with 0 fatal errors
   ```

---

## Common Pitfalls
- Using theoretical theoretical memory bandwidth (504 GB/s) rather than measured sustained bandwidth (255.8 GB/s).
- Plotting on linear rather than logarithmic axes, obscuring memory-bandwidth-bound knee points.
- Omitting the origin citation to `AUDIT_03` in the figure caption.

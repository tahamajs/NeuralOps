---
name: experimental-design
description: "Design controlled multi-factor experiments, ablation studies, and parameter sweeps for NeuralOps"
version: "1.0.0"
tags: ["neuralops", "experiment", "ablation", "methodology", "design"]
---

# Skill: Experimental Design and Ablation Matrix Construction

## When to use
Use this skill when designing rigorous ablation studies, low-rank parameter sweeps, and multi-pillar distillation matrices for NeuralOps evaluation. It structures multi-factor experimental configurations ensuring proper baseline controls, isolated variables, and reproducible ledger logging.

## Prerequisites
- Manuscript context: `thesis_final/810101504_main.tex`.
- Remote GPU host: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) with Python path `/mnt/d/venv/bin/python3`.
- Master results ledger: `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
All experiment designs must align with the empirical methodologies established in the canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced for the 5-stage ablation design and KD Regimes 0 through 5 loss matrix.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced for rank-sweep protocols ($r \in [16, 32, 64, 128]$) and Cholesky whitening factors.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced for standard benchmark configurations ($B=16, S=2048$, cold L2 cache flushes).
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced for sequence length sweep parameters $N \in [128, 8192]$.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced for testbed hardware disclosure requirements and commit tracking.

## Canonical Repository Invariants
Experiments must maintain consistency with canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Experimental Design Methodology

### 1. Factorial Matrix Design
When evaluating architectural modifications or distillation components:
- **Single-Variable Isolation**: Vary exactly one architectural factor at a time (e.g., adding spectral loss while holding learning rate and calibration steps constant).
- **Frozen Baseline Controls**: Always evaluate against uncompressed reference models and unweighted baseline truncation (Regime 0).
- **Fixed Random Seeds**: Guarantee deterministic weight initialization and batch sequencing across all comparative regimes.

### 2. The 6-Pillar Distillation Ablation Protocol
Structure distillation ablation sweeps across the 6 canonical loss pillars:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{KL}} + \gamma \mathcal{L}_{\text{MSE}} + \delta \mathcal{L}_{\text{cos}} + \epsilon \mathcal{L}_{\sigma} + \zeta \mathcal{L}_{\text{cov}}$$
- Regime 0: $\alpha=1.0$ (Truncated unweighted baseline).
- Regime 1: $+\beta \mathcal{L}_{\text{KL}}$ (Logit transfer).
- Regime 2: $+\gamma \mathcal{L}_{\text{MSE}}$ (Hidden state feature alignment).
- Regime 3: $+\delta \mathcal{L}_{\text{cos}}$ (Attention map cosine alignment).
- Regime 4: $+\epsilon \mathcal{L}_{\sigma}$ (Singular value spectral preservation).
- Regime 5: $+\zeta \mathcal{L}_{\text{cov}}$ (Covariance and contrastive alignment).

### 3. Ledger Logging Protocol
Every experimental trial must be registered into `results/registry/master_ledger.jsonl`:
```python
from neuralops.core.results_ledger import log_experiment_result

log_experiment_result(
    module_name="hnc_svd_rank64",
    category="compressor",
    family="low_rank",
    experimental_setup={"hardware": "RTX 4070", "rank": 64, "batch_size": 16},
    metrics={"accuracy": 73.62, "speedup": 2.14, "latency_ms": 0.42}
)
```

---

## Verification
Confirm the experimental design meets quality gates:
1. **Canonical Invariants**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must return 131 / 72 / 26
   ```
2. **Audit Alignment**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Compilation Check**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must produce 0 fatal errors
   ```

---

## Common Pitfalls
- Confounding multiple hyperparameters simultaneously (e.g., changing rank and learning rate together).
- Skipping the unweighted Regime 0 baseline, making relative gains impossible to isolate.
- Omitting hardware warmups or cold cache flushes, introducing timing jitter into latency sweeps.

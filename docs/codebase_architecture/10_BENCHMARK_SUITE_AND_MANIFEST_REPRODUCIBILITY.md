# NeuralOps Codebase Architecture: Part 10
# Benchmark Suite, Master Ledger & Manifest Reproducibility

**Assigned Subagent**: `Agent 10: Benchmark Auditor & Manifest Reproducibility Specialist`  
**Target Codebase Paths**:
- `neuralops/benchmarks/`
- `neuralops/results_saver/`
- `neuralops/validation/`
- `scripts/`

---

## 1. Executive Summary & Design Philosophy

Empirical claims in modern deep learning research frequently suffer from reproducibility failure: non-versioned datasets, undisclosed random seeds, unrecorded driver versions, and manual post-processing errors.

Part 10 details the **Benchmark Suite & Provenance Ledger Architecture** in `neuralops/results_saver/` and `neuralops/benchmarks/`:
- **SurgeryManifest**: Every evaluation experiment cryptographically binds the Git commit SHA, CUDA version, PyTorch version, hardware ID, random seeds, and parameter hashes.
- **Master SQLite / JSONL Ledger**: Results are automatically appended to an immutable master ledger (`results/registry/master_ledger.jsonl`), completely eliminating manual copy-pasting of performance numbers.
- **3-Tier Release Quality Gates**: Pre-merge validation enforcing:
  1. `PASS_FAST` (CPU logic, AST dead-code audit, lint checks).
  2. `PASS_SLOW` (End-to-end task evaluation on WikiText-2, PTB, GLUE).
  3. `PASS_HARDWARE` (Cross-backend numerical parity and gradcheck).
- **Automated Artifact Generation**: Automatically compiles ledger records directly into LaTeX `booktabs` tables and Pareto frontier plots.

---

## 2. Assigned Subagent Responsibilities (Agent 10)

Agent 10 is responsible for:
1. **Ledger Immutability**: Ensuring that ledger writers use atomic append operations with file locking to prevent corrupted JSONL entries during parallel sweeps.
2. **Quality Gate Verification**: Running and enforcing `scripts/utils/audit_dead_code.py` and `scripts/audit_readme_consistency.py` with zero tolerance for dead code or count drift.
3. **LaTeX Exporter Accuracy**: Verifying that exported LaTeX tables match master ledger values bit-for-bit without manual intervention or approximation.
4. **Dataset Ingestion Pipeline**: Ensuring standard loaders for WikiText-2, PTB, and GLUE handle token padding, document boundaries, and `cu_seqlens` correctly.

---

## 3. Data Flow & Provenance Architecture

```
[Benchmark Task Execution (e.g. WikiText-2)]
                     │
                     ▼
       [Hardware & Telemetry Probe]
(Record latency, VRAM, GPU temperature, commit SHA)
                     │
                     ▼
          [SurgeryManifest Creation]
  (Cryptographic signature of experimental setup)
                     │
                     ▼
        [Atomic Ledger Writer Engine]
     ├── Append to master_ledger.jsonl (with fcntl lock)
     └── Sync to results_ledger.sqlite
                     │
                     ▼
     [Automated Report & Table Exporters]
     ├── LaTeX booktabs tables -> academic/thesis/tables/
     └── Matplotlib / TikZ figures -> academic/thesis/figures/
```

---

## 4. Implementation Blueprint: SurgeryManifest Schema

```python
# In neuralops/results_saver/manifest.py
@dataclass
class SurgeryManifest:
    manifest_id: str
    timestamp: str
    git_commit_sha: str
    device_name: str
    cuda_driver_version: str
    torch_version: str
    random_seed: int
    operator_name: str
    compression_ratio: float
    perplexity: float
    latency_ms: float
    peak_vram_mb: float
    parameter_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write_atomic(self, file_path: str):
        with open(file_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(asdict(self)) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
```

---

## 5. Audit & Validation Standards

- **Test Suite**: `tests/benchmarks/test_manifest_ledger.py`
- Verification assertions:
  1. Ledger write atomicity under 16 concurrent processes.
  2. Zero mismatch between `master_ledger.jsonl` and SQLite mirror.
  3. Strict zero-error result on `python3 scripts/utils/audit_dead_code.py`.

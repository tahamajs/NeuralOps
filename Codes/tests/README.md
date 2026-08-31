# Test Suite Subsystem (`Codes/framework/tests`)

The `tests` subsystem provides a comprehensive pytest test harness verifying the mathematical correctness, numerical stability, zero-shot fidelity, gradient flow, and hardware performance of all framework components.

---

## 🏗 Test Organization

- **Attention Mechanism Tests (`test_attention_modules.py`, `test_attentions.py`)**: Validates shape correctness $, numerical finiteness (no NaN/Inf), gradient flow, padding mask compliance, multi-head reshaping, and zero-shot weight transfer across all registered attention variants.
- **Attention Injection Engine (`test_attention_injection.py`)**: Verifies universal layer replacement on real HuggingFace transformer architectures with both `copy_weights=True` and `copy_weights=False`.
- **Fused Kernel Verification (`test_fused_svd_kernel.py`)**: Analytical gradient checks (`gradcheck`), forward/backward numerical parity against un-fused references, and RMSNorm integration.
- **Compression & SVD Specialist Suites (`test_compressors.py`, `test_all_compression_engines.py`, `test_svd_audit_suite.py`)**: Verifies SVD truncation, rank selection algorithms, and weight reconstruction fidelity.
- **Dataset & Trainer Verification (`test_datasets.py`, `test_results_saver.py`, `test_multi_mode.py`)**: Checks data loaders, tokenizer collators, trainer factory resolution, and artifact serialization.

---

## 📦 Key Test Files

| Test File | Target Subsystem / Verification Scope |
|:---|:---|
| `test_attention_injection.py` | Universal attention injection into DistilBERT, BERT, GPT-2 models. |
| `test_attention_modules.py` | Exhaustive 7-point verification suite across all attention modules. |
| `test_fused_svd_kernel.py` | Fused SVD linear forward/backward numerical stability and gradchecks. |
| `test_all_compression_engines.py` | Verification of spectral, pruning, and low-rank compression algorithms. |
| `test_results_saver.py` | Multi-backend serialization (JSON, Parquet, checkpoints). |
| `test_datasets.py` | GLUE dataset loading, tokenization pipelines, and synthetic generator. |
| `conftest.py` | Shared pytest fixtures (model configs, hidden states, attention masks). |

---

## 🚀 Running the Test Suite

### 1. Run Fast Core Tests
```bash
pytest Codes/framework/tests/test_attention_injection.py Codes/framework/tests/test_results_saver.py Codes/framework/tests/test_fused_svd_kernel.py -v
```

### 2. Run Attention Mechanism Tests
```bash
pytest Codes/framework/tests/test_attention_modules.py -v -k "svd or linear or full"
```

### 3. Run All Integration Tests
```bash
pytest Codes/framework/tests/ -m "integration" -v
```

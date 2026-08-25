# 🔬 NeuralOps Framework — EXHAUSTIVE Remediation & Completion Plan

> **Generated**: 2026-08-25 from 5-agent deep parallel audit  
> **Scope**: Every .py module, every .md doc, every subsystem  
> **Goal**: Make framework fully conference-ready for NeurIPS/ICLR/ICML submission

---

## TABLE OF CONTENTS

1. [Executive Audit Summary](#1-executive-audit-summary)
2. [PHASE 1: CRITICAL — Fake Data & Broken Modules](#2-phase-1-critical)
3. [PHASE 2: HIGH — Code Quality & Safety](#3-phase-2-high)
4. [PHASE 3: MEDIUM — Cleanup & Dead Code](#4-phase-3-medium)
5. [PHASE 4: Infrastructure & Build System](#5-phase-4-infrastructure)
6. [PHASE 5: Documentation Complete Rewrite](#6-phase-5-documentation)
7. [Per-File Action Registry — Attentions](#7-per-file-attentions)
8. [Per-File Action Registry — Compressions](#8-per-file-compressions)
9. [Per-File Action Registry — Core](#9-per-file-core)
10. [Per-File Action Registry — Infrastructure](#10-per-file-infrastructure)
11. [Per-File Action Registry — Documentation](#11-per-file-documentation)
12. [Execution Timeline & Dependencies](#12-execution-timeline)

---

## 1. Executive Audit Summary

### What Was Audited
- **236 registered attention mechanisms** in `Codes/framework/attentions/`
- **107 registered compression engines** in `Codes/framework/compression_engine/`
- **695 total Python files** across the entire framework
- **400+ markdown documentation files** in `Docs/content/`
- **15 infrastructure subsystems** (trainer, data, visualization, analysis, tests, etc.)
- **5 evaluation tracks** mapped in `eval_all_models_datasets.py`

### Critical Findings Summary

| Category | Finding | Severity | Impact |
|:---|:---|:---:|:---|
| **Fake Benchmarks** | 50+ attention docs and 30+ compressor docs have IDENTICAL benchmark numbers | 🔴 CRITICAL | Academic misconduct risk — instant rejection |
| **Fake Eval Data** | `eval_all_models_datasets.py` uses `torch.randint` synthetic data for ALL evaluations | 🔴 CRITICAL | All reported results are meaningless |
| **Fake Validation Reports** | `spectral_ssm_validation_report.md` shows 30 models with identical 50.92% accuracy | 🔴 CRITICAL | Fabricated results |
| **Fake Blog Posts** | 34 blog posts are word-for-word identical (only module name swapped) | 🔴 CRITICAL | Academic misconduct risk |
| **Stub Modules** | 7+ attention files and 7+ compressor files are <50 lines stubs | 🔴 CRITICAL | Broken functionality |
| **Missing .weight** | Multiple attention modules crash T5 due to missing `@property def weight` | 🔴 CRITICAL | Runtime crashes |
| **Silent Exceptions** | 20+ `except Exception: pass` blocks across core files | 🟠 HIGH | Hidden bugs |
| **Hardcoded Dims** | `hidden_size=768`, `num_heads=12`, `num_labels=2` hardcoded in core | 🟠 HIGH | Fails on non-BERT models |
| **Bloated READMEs** | 15 subdirectory READMEs are identical ~3000-line auto-generated templates | 🟡 MEDIUM | Reviewer red flag |
| **No Build System** | No `setup.py` or `pyproject.toml` — `pip install -e` fails | 🟡 MEDIUM | Not installable |
| **Stub Trainers** | `rl_trainer.py`, `unsloth_trainer.py` raise NotImplementedError | 🟡 MEDIUM | Dead code |
| **Template Files** | `replaceme_compressor.py`, `your_attention_name.md` exist in production | 🟡 MEDIUM | Unprofessional |

---

## 2. PHASE 1: CRITICAL — Fake Data & Broken Modules

### 1.1 FAKE EVALUATION DATA IN eval_all_models_datasets.py

**File**: `Codes/framework/eval_all_models_datasets.py` (~692 lines)

**Problem**: The function `generate_dataset_batch()` (lines ~202-242) generates ALL evaluation data using `torch.randint()` regardless of which dataset is requested. This means:
- SST-2 "accuracy" is measured on random tokens
- MRPC "F1" is measured on random tokens
- ALL benchmark numbers across the framework are fabricated

**Exact problematic code pattern**:
```python
def generate_dataset_batch(dataset_name, model_name, batch_size=8, seq_len=128):
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))  # FAKE DATA
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,))  # RANDOM LABELS
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

**Required Fix**:
1. Replace `generate_dataset_batch()` with actual dataset loading from `data/glue_loader.py` or `data/prepare_datasets.py`
2. Load real tokenized GLUE splits (SST-2, MRPC, QNLI, etc.)
3. Use the correct tokenizer for each model (BERT tokenizer for BERT, RoBERTa tokenizer for RoBERTa, T5 tokenizer for T5)
4. Only fall back to synthetic data with an explicit `--synthetic` flag and clear warning

**Also affects**: `core/multi_mode_benchmark.py` line ~150 — same fake data pattern.

### 1.2 FAKE BENCHMARK DOCUMENTATION

**Problem**: The `enrich_module_docs.py` script (which we ran earlier) injected IDENTICAL benchmark tables into 80+ documentation files. Every module claims the exact same accuracy, speedup, and memory reduction:
- BERT SST-2: 83.65% (attentions) or 83.70% (compressors)
- BERT MRPC: 87.90% or 87.95%
- RoBERTa SST-2: 94.20% or 94.35%
- All speedups: 2.14x, 2.08x, 2.21x, etc.

A conference reviewer will immediately notice 50+ modules all claiming identical performance.

**Files affected** (by category):
- **Attention docs with fake benchmarks**: ~88+ files in `Docs/content/attentions/`
- **Compression docs with fake benchmarks**: ~59+ files in `Docs/content/compressions/`
- **Blog posts (word-for-word identical)**: 34 files in `Docs/content/compressions_blog/`

**Required Fix**:
1. Remove ALL auto-generated benchmark tables from every .md file
2. Replace with "Benchmarks pending — run on GPU server with: `bash run_on_gpu_server.sh --mode comprehensive`"
3. After running actual GPU benchmarks, insert REAL per-module numbers
4. Each module doc must have UNIQUE mathematical formulations specific to that algorithm

### 1.3 FAKE VALIDATION REPORTS

**Files**:
- `Docs/content/spectral_ssm_validation_report.md` — 30 different models all show exactly 50.92% accuracy and 33.74% F1
- `Docs/content/svd_lowrank_validation_report.md` — Similar fabricated uniformity
- `Docs/content/moe_pruning_validation_report.md` — Template results
- `Docs/content/kernel_quantization_validation_report.md` — Template results

**Required Fix**: Delete all 4 reports. Regenerate from actual GPU benchmark runs.

### 1.4 STUB/EMPTY ATTENTION MODULES

These attention files are stubs (<50 lines, often just re-exports or empty classes):

| File | Lines | Issue | Action Required |
|:---|:---:|:---|:---|
| `fasa.py` | <30 | Empty stub, no real implementation | Implement fully or remove from registry |
| `cur_moe.py` | <30 | Stub re-export | Implement fully or remove |
| `svd_linear_decay.py` | <30 | Stub | Implement or remove |
| `salt.py` | <30 | Stub | Implement or remove |
| `moba.py` | <30 | Stub, recently created | Implement or remove |
| `svd_llm_v2.py` | <30 | Stub re-export | Implement or remove |
| `projection_factorized_attn.py` | <30 | Stub | Implement or remove |

### 1.5 STUB/EMPTY COMPRESSION MODULES

| File | Lines | Issue | Action Required |
|:---|:---:|:---|:---|
| `replaceme_compressor.py` | <20 | Literal template file — class `ReplaceMeCompressor` | DELETE immediately |
| `aasvd.py` | ~10 | Alias stub for `aa_svd_compressor.py` | Consolidate |
| `adasvd.py` | ~10 | Alias stub for `adasvd_compressor.py` | Consolidate |
| `mgie_svd.py` | ~10 | Alias stub | Consolidate |
| `saes_svd.py` | ~10 | Alias stub | Consolidate |
| `cur_moe.py` | ~10 | Alias stub | Consolidate |
| `grecodec.py` | ~10 | Alias stub | Consolidate |
| `frandi.py` | ~10 | Alias stub | Consolidate |

### 1.6 MISSING @property def weight (T5 CRASH)

T5's encoder introspects child modules for `.weight` during forward pass. Any custom module replacing `nn.Linear` that lacks `@property def weight` will crash T5 with `AttributeError`.

**Confirmed affected files**:
| File | Class | Status |
|:---|:---|:---|
| `SOAFusedAttention.py` | Custom linear layers | ❌ Missing .weight |
| `aegis_hybrid.py` | Custom projection | ❌ Missing .weight |
| `aegis_v1.py` | Custom projection | ❌ Missing .weight |
| `SVD-Mamba.py` | Factorized layers | ❌ Missing .weight |

**Files confirmed OK** (already have .weight):
- `svd_linear_attention.py` ✅
- `adaptive_svd_attention.py` ✅
- `fused_svd_kernel.py` ✅
- `asvd_attention.py` ✅

**Required Fix**: Audit EVERY attention and compression module that defines a custom `nn.Module` replacing `nn.Linear`. Add:
```python
@property
def weight(self) -> torch.Tensor:
    return torch.matmul(self.U, self.V)  # or equivalent reconstruction
```


---

## 3. PHASE 2: HIGH — Code Quality & Safety

### 2.1 SILENT EXCEPTION SWALLOWING (except Exception: pass)

This is a systematic anti-pattern across the codebase. Every `except Exception: pass` hides real bugs. Here is every instance found:

| File | Line(s) | Code Pattern | Risk |
|:---|:---:|:---|:---|
| `core/registry.py` | L50, L59 | `except Exception: pass` in `_try_lazy_load()` | Hides import errors in attention/compression modules |
| `core/base.py` | L32, L111 | Abstract methods use `pass` instead of `raise NotImplementedError()` | Subclasses can silently skip implementation |
| `core/base.py` | L91 | `except Exception: pass` in `copy_weights()` | Weight adaptation failures silently ignored |
| `core/attention_utils.py` | L320 | `except ValueError: pass` | Failed attention injection silently ignored |
| `core/attention_utils.py` | L350 | `except Exception: pass` | Constructor failures silently ignored |
| `core/attention_utils.py` | L358 | Generic exception handler | Error masking |
| `core/memory.py` | L34, L57 | `except Exception: pass` in MPS memory clearing | Memory leaks on Apple Silicon |
| `auto_run.py` | L138 | `except Exception: continue` reading JSON | Skips malformed results silently |
| `check_framework_sync.py` | L84 | `except Exception: pass` in dummy linear test | Hides instantiation failures |
| `eval_all_models_datasets.py` | L308-316 | Nested `except Exception:` blocks | Brute-force class instantiation |

**Required Fix for each**: Replace `except Exception: pass` with:
```python
except ImportError as e:
    logger.warning(f"Failed to import module: {e}")
except (ValueError, RuntimeError) as e:
    logger.error(f"Module initialization failed: {e}")
    raise
```

### 2.2 HARDCODED MAGIC NUMBERS

| File | Line(s) | Hardcoded Value | Problem | Fix |
|:---|:---:|:---|:---|:---|
| `core/attention_utils.py` | L266-285 | `hidden_size=768, num_heads=12` | Fails for LLaMA (4096), GPT-2 (768/12 OK but others not) | Read from model.config |
| `core/model_wrapper.py` | L192 | `num_labels=2` | Binary classification only | Accept as parameter |
| `compression_engine/direct_svd.py` | Multiple | `768.0` reference dimension | Rank scaling breaks for dim≠768 | Use `model.config.hidden_size` |
| `attentions/aegis_v1.py` | Multiple | `clamp(max=11.0)`, `-10000.0` mask | Precision-dependent magic | Use `torch.finfo(dtype).min` |
| `attentions/bam_umoe.py` | Multiple | `noise_std=0.1` | Not configurable | Add constructor parameter |
| `attentions/mamba_svd_flash.py` | Multiple | Triple `.clamp(min=1e-6)` | Redundant operations | Single clamp is sufficient |

### 2.3 FP16/BF16 NUMERICAL OVERFLOW RISK

**File**: `compression_engine/aa_svd_compressor.py`

**Problem**: The Activation-Aware SVD computes streaming EMA covariance as `(X^T @ X) / B` in the current dtype. When running in FP16/BF16 with large batch sizes or long sequences, this produces `Inf`/`NaN` values because:
- FP16 max value is 65504
- A single hidden activation of magnitude 10 squared gives 100, times seq_len=4096 = 409600 which overflows FP16

**Required Fix**:
```python
# BEFORE (dangerous):
cov = (X.transpose(-2, -1) @ X) / batch_size

# AFTER (safe):
with torch.cuda.amp.autocast(enabled=False):
    X_fp32 = X.float()
    cov = (X_fp32.transpose(-2, -1) @ X_fp32) / batch_size
```

### 2.4 BRITTLE 9-SIGNATURE CONSTRUCTOR PROBING

**File**: `core/attention_utils.py` lines L293-303

**Problem**: When creating attention module instances, the code tries up to 9 different constructor signatures in nested try/except blocks:
```python
try:
    module = attn_cls(config)
except:
    try:
        module = attn_cls(config, rank=rank)
    except:
        try:
            module = attn_cls(hidden_size, num_heads)
        except:
            # ... 6 more attempts
```

This is extremely fragile and hides real constructor errors.

**Required Fix**: Standardize ALL attention module constructors to accept:
```python
def __init__(self, config, rank: int = 32, **kwargs):
```
Then use a single instantiation call with error propagation.

---

## 4. PHASE 3: MEDIUM — Cleanup & Dead Code

### 3.1 BLOATED TEMPLATE READMEs (15 directories)

Every subdirectory contains an identical ~3000-line, ~111KB "Exhaustive Masterclass Technical Specification" README. From Section 4 onwards, they contain copy-pasted dummy classes (`FrameworkCoreEngine_01` through `FrameworkCoreEngine_09`) with identically repeated boilerplate code. This is artificially padded content.

**All affected directories** (each needs a rewrite to 50-100 lines):

| Directory | Current README Size | Content | Action |
|:---|:---:|:---|:---|
| `core/` | ~111KB, 2975 lines | Fake class padding | Rewrite: describe registry, base classes, injection API |
| `trainer/` | ~111KB | Template | Rewrite: describe HF trainer, custom loop, DDP |
| `data/` | ~111KB | Template | Rewrite: describe prepare_datasets, glue_loader |
| `visualization/` | ~111KB | Template | Rewrite: describe chart_generator, heatmaps |
| `analysis/` | ~111KB | Template | Rewrite: describe layerwise analyzer |
| `results_saver/` | ~111KB | Template | Rewrite: describe JSON/Parquet/SQLite exporters |
| `tests/` | ~111KB | Template | Rewrite: describe test suites, how to run |
| `distributed/` | ~111KB | Template | Rewrite: describe DDP/FSDP strategies |
| `config/` | ~111KB | Template | Rewrite: describe config dataclasses |
| `utils/` | ~111KB | Template | Rewrite: describe utility functions |
| `router/` | ~111KB | Template | Rewrite: describe ML/keyword/ensemble routers |
| `drift/` | ~111KB | Template | Rewrite: describe drift detectors |
| `tracking/` | ~111KB | Template | Rewrite: describe experiment trackers |
| `experiment_logging/` | ~111KB | Template | Rewrite: describe loggers |
| `models_conf/` | ~111KB | Template | Rewrite: describe model configs |
| `web/` | ~1.2KB | ✅ Accurate | Keep as-is |

### 3.2 REDUNDANT CODE

| File | Line(s) | Issue | Fix |
|:---|:---:|:---|:---|
| `attentions/mamba_svd_flash.py` | Multiple | `.clamp(min=1e-6).clamp(min=1e-6).clamp(min=1e-6) * 0.95` | Single `.clamp(min=1e-6) * 0.95` |
| `attentions/SOAFusedAttention.py` | L75-76 | `S = S.clamp(min=1e-6); S = S.clamp(min=1e-6)` | Single clamp |

### 3.3 ALIAS FILE CLUTTER IN COMPRESSION ENGINE

Many compression modules have TWO files: a full implementation and a ~10-line alias stub. This creates confusion and maintenance burden.

| Alias Stub File | Points To (Real Implementation) | Action |
|:---|:---|:---|
| `aasvd.py` | `aa_svd_compressor.py` | Remove alias, add registry alias in real file |
| `adasvd.py` | `adasvd_compressor.py` | Remove alias |
| `mgie_svd.py` | `mgie_svd_compressor.py` | Remove alias |
| `saes_svd.py` | `saes_svd_compressor.py` | Remove alias |
| `grecodec.py` | `grecodec_compressor.py` | Remove alias |
| `frandi.py` | `frandi_compressor.py` | Remove alias |
| `cur_moe.py` (compressor) | `cur_moe_compressor.py` | Remove alias |

### 3.4 TEMPLATE/DEAD FILES TO DELETE

| File | Reason to Delete |
|:---|:---|
| `compression_engine/replaceme_compressor.py` | Literal template class `ReplaceMeCompressor` |
| `Docs/content/attentions/your_attention_name.md` | Template doc |
| `Docs/content/compressions/your_compressor_name.md` | Template doc |
| `Docs/content/attentions/zeros.md` | Placeholder |

### 3.5 STUB IMPLEMENTATIONS TO COMPLETE OR REMOVE

| File | Class/Function | Issue | Action |
|:---|:---|:---|:---|
| `trainer/rl_trainer.py` | `RLTrainer` | Raises `NotImplementedError` | Implement or remove |
| `trainer/unsloth_trainer.py` | `UnslothTrainer` | Raises `NotImplementedError` | Implement or remove |
| `visualization/plotter.py` | Plotting functions | Raises `NotImplementedError` | Implement or remove |
| `models_conf/variants/export/tflite.py` | TFLite export | Raises `NotImplementedError` | Implement or remove |
| `tracking/tracker.py` | Unsupported backends | Raises `NotImplementedError` | Document supported backends |

### 3.6 PRINT STATEMENTS → LOGGING

| File | Line(s) | Pattern | Fix |
|:---|:---:|:---|:---|
| `core/compression_engine.py` | L32, L41, L48 | `print()` | `logger.info()` |
| `parallel_track_checker.py` | L114-117, L152-166 | `print()` | `logger.info()` |
| `launch_gpu_server.py` | L131-148 | `print()` | `logger.info()` |

### 3.7 TODO/FIXME/RESEARCH FIX COMMENTS

Found in compression engine files:
| File | Comment | Priority |
|:---|:---|:---:|
| `aa_svd_compressor.py` | `# FIX: Comprehensive multi-domain calibration corpus (>512 tokens)` | HIGH |
| `aa_svd_compressor.py` | `# RESEARCH FIX: Accelerated fused kernel dispatch` | MEDIUM |
| `dynamic_rank.py` | `# RESEARCH FIX: Robust empirical singular entropy computation with Tikhonov damping` | HIGH |
| `res_svd.py` | `# RESEARCH FIX: Analytical spectral initialization of LoRA adapter weights (A, B)` | MEDIUM |


---

## 5. PHASE 4: Infrastructure & Build System

### 4.1 NO PACKAGE BUILD SYSTEM

**Problem**: The root README claims `pip install -e Codes/framework/` works, but there is no `setup.py`, `pyproject.toml`, or `setup.cfg` in the framework directory.

**Required Fix**: Create `Codes/framework/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "neuralops-framework"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.38.0",
    "datasets>=2.18.0",
    "tokenizers>=0.15.0",
    "accelerate>=0.27.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.8.0",
    "tqdm>=4.66.0",
]
```

### 4.2 TEST COVERAGE GAPS

**Problem**: Some tests in `tests/` use `pytest.skip` dynamically due to missing imports or pretrained weights. Tests run on synthetic data.

**Files with pytest.skip**:
- `tests/test_attention_injection.py`
- `tests/test_attention_modules.py`

**Required Fix**:
1. Fix import paths so tests don't need to skip
2. Add lightweight integration tests using small subsets of real GLUE data
3. Create a `conftest.py` with shared fixtures for tiny model configs

### 4.3 MISSING REQUIREMENTS FILES

**Check needed**: Verify `Codes/django_api/requirements.txt` exists and is complete. Create `Codes/framework/requirements.txt` if missing.

---

## 6. PHASE 5: Documentation Complete Rewrite

### 6.1 DOCUMENTATION CATEGORIES & STATUS

| Category | Total Files | Genuine | Auto-Generated | Action |
|:---|:---:|:---:|:---:|:---|
| Attention module docs (`Docs/content/attentions/`) | ~236 | ~15 well-written | ~221 template/fake | Rewrite top 20, mark rest as pending |
| Compression module docs (`Docs/content/compressions/`) | ~107 | ~10 well-written | ~97 template/fake | Rewrite top 20, mark rest as pending |
| Compression blog posts (`Docs/content/compressions_blog/`) | ~68 | ~5 unique | ~34 word-for-word identical | Delete duplicates, keep unique ones |
| Subdirectory READMEs | 15 | 1 (web/) | 14 identical ~111KB templates | Rewrite each to 50-100 lines |
| Validation reports | 4 | 0 | 4 fabricated | Delete, regenerate from real GPU runs |
| Root READMEs | 4 | 4 (recently updated) | 0 | ✅ OK |

### 6.2 GENUINE VS AUTO-GENERATED DOCUMENTATION

**Examples of GENUINE well-written docs** (keep and use as templates):
- `Docs/content/compressions/direct_svd.md` — Genuinely detailed, mathematically rigorous, module-specific
- `Docs/content/attentions/svd_linear_attention.md` — Real mathematical detail
- `Docs/content/attentions/performer_attention.md` — Algorithm-specific FAVOR+ formulations

**Examples of AUTO-GENERATED template docs** (need rewrite):
- Any file with these exact numbers: `83.65%`, `83.70%`, `2.14x`, `87.90%`
- Any file with the exact string `PYTHONPATH=Codes python3 -c "` on line 99
- Any file where the "Mathematical Formulation" section uses generic SVD equations identical to 50+ other files

### 6.3 WHAT GENUINE MODULE DOCUMENTATION MUST CONTAIN

For each module to pass conference peer review:

1. **Module-Specific Algorithm Description**: NOT generic "factorized latent projections" — describe what THIS specific algorithm does differently
2. **Unique Mathematical Formulation**: The Performer uses FAVOR+ random features ($\phi(x) = \exp(x^T\omega - \|x\|^2/2)$), NOT generic SVD. The Nyström method uses landmark sampling. Each must be different.
3. **Specific Complexity Analysis**: Performer is $O(N \cdot r \cdot D)$ where $r$ is the number of random features. Nyström is $O(N \cdot m \cdot D)$ where $m$ is landmarks. These are DIFFERENT.
4. **Real Benchmark Numbers**: From actual GPU runs, or explicitly marked as "pending"
5. **Unique Code Example**: Showing the specific API of THAT module, not a generic `inject_attention` call
6. **Known Limitations**: Every algorithm has trade-offs — document them honestly
7. **References**: Cite the original paper for each algorithm

### 6.4 TOP 20 MODULES REQUIRING PRIORITY DOCUMENTATION

These are the most important modules that need genuine, hand-crafted documentation:

| # | Module | Type | Paper | Why Priority |
|:---:|:---|:---|:---|:---|
| 1 | `svd_linear_attention` | Attention | Original contribution | Flagship module |
| 2 | `adaptive_svd_attention` | Attention | Original contribution | Core variant |
| 3 | `direct_svd` | Compressor | Standard SVD | Foundation compressor |
| 4 | `aa_svd_compressor` | Compressor | AASVD (Activation-Aware) | Key innovation |
| 5 | `erc_svd` | Compressor | Error-Controlled SVD | Key innovation |
| 6 | `performer_attention` | Attention | Choromanski et al. 2021 | FAVOR+ kernel |
| 7 | `flash_attention3` | Attention | Dao 2023 | Industry standard |
| 8 | `nystrom_attention` | Attention | Xiong et al. 2021 | Nyström landmark |
| 9 | `cosformer` | Attention | Qin et al. 2022 | Cosine re-weighting |
| 10 | `aegis_v1` | Attention | Original contribution | AEGIS family lead |
| 11 | `slicegpt` | Compressor | Ashkboos et al. 2024 | Rotational pruning |
| 12 | `sparsegpt` | Compressor | Frantar & Alistarh 2023 | Hessian pruning |
| 13 | `admm_pruner` | Compressor | Structured pruning | ADMM optimization |
| 14 | `delta_kv` | Compressor | KV cache | Differential compression |
| 15 | `bam_umoe` | Attention | MoE routing | Mixture-of-Experts |
| 16 | `mamba_svd_flash` | Attention | SSM hybrid | State-space model |
| 17 | `gated_linear_attention` | Attention | Linear attention | Gated recurrence |
| 18 | `kronecker_tt_lora` | Compressor | Tensor decomposition | Kronecker factorization |
| 19 | `astro_quantizer` | Compressor | Quantization | Post-training quant |
| 20 | `entropy_budget_allocation` | Compressor | Entropy budgeting | Rank allocation |

---

## 7. Per-File Action Registry — Attention Modules

### Status Legend
- ✅ **OK**: Full implementation, no issues
- 🔧 **CLEANUP**: Minor issues (dead code, magic numbers, style)
- ⚠️ **FIX**: Missing .weight, broken forward(), needs repair
- ❌ **STUB**: Empty/alias, needs full implementation or removal
- 📝 **DOC**: Documentation is auto-generated template, needs rewrite

### Track 1: SVD & Low-Rank (13 Attentions)
| File | Status | Code | Doc | Issues |
|:---|:---:|:---:|:---:|:---|
| `svd_linear_attention.py` | ✅ | OK | 📝 | Doc needs unique math |
| `adaptive_svd_attention.py` | ✅ | OK | 📝 | Doc needs unique math |
| `asvd_attention.py` | ✅ | OK | 📝 | Doc needs unique math |
| `linformer_attention.py` | ✅ | OK | 📝 | Doc template |
| `fused_svd_kernel.py` | ✅ | OK | 📝 | Doc template |
| `svd_projection_attention.py` | 🔧 | Minor | 📝 | Check .weight |
| `svd_llm_v2_attention.py` | ⚠️ | Check | 📝 | Verify not stub |
| `factorised_matrix_state.py` | 🔧 | Minor | 📝 | Doc template |
| `res_svd.py` | 🔧 | FIX comment | 📝 | Has RESEARCH FIX comment |

### Track 2: Spectral, SSM & Recurrent (28 Attentions)
| File | Status | Issues |
|:---|:---:|:---|
| `aegis_v1.py` | ⚠️ | Missing .weight, hardcoded clamp(max=11.0) |
| `aegis_v2.py` through `aegis_v6.py` | 🔧 | Check .weight on each |
| `aegis_hybrid.py` | ⚠️ | Missing .weight |
| `SVD-Mamba.py` | ⚠️ | Missing .weight |
| `mamba_svd_flash.py` | 🔧 | Triple redundant clamp |
| `gated_delta_net.py` | 🔧 | Check doc |
| `gated_deltanet_block.py` | 🔧 | Doc missing benchmarks/proofs |
| `hgrn2_layer.py` | 🔧 | Check |
| `griffin_ssd.py` | 🔧 | Check |
| `spectron.py` | 🔧 | Check |
| `galerkin_attention.py` | 🔧 | Check |
| `vspec_attention.py` | 🔧 | Doc missing proofs |

### Track 3: MoE, Hybrid & Routing (20 Attentions)
| File | Status | Issues |
|:---|:---:|:---|
| `bam_umoe.py` | 🔧 | Hardcoded noise_std=0.1 |
| `fasa.py` | ❌ | STUB — needs full implementation |
| `cur_moe.py` (attention) | ❌ | STUB |
| `moba.py` | ❌ | STUB |
| `salt.py` | ❌ | STUB |
| `projection_factorized_attn.py` | ❌ | STUB |
| `samba_moe.py` | 🔧 | Check completeness |
| `jamba.py` | 🔧 | Check completeness |
| `hymba_transx.py` | 🔧 | Check completeness |

### Track 4: Kernel Approximations (11 Attentions)
| File | Status | Issues |
|:---|:---:|:---|
| `performer_attention.py` | ✅ | OK, doc needs unique FAVOR+ math |
| `nystrom_attention.py` | ✅ | OK, doc needs landmark math |
| `flash_attention3.py` | ✅ | OK |
| `cosformer.py` | ✅ | OK, doc needs cosine reweighting math |
| `SOAFusedAttention.py` | ⚠️ | Missing .weight, redundant clamp |
| `cluster_attention.py` | 🔧 | Check |
| `adasplash_attention.py` | ✅ | OK |

### Track 5: Pruning-Related (2 Attentions)
| File | Status | Issues |
|:---|:---:|:---|
| `linformer_attention.py` | ✅ | OK |
| `adaptive_sparse_attention.py` | 🔧 | Check |

---

## 8. Per-File Action Registry — Compression Modules

### Track 1: SVD-Based (17 Compressors)
| File | Status | Issues |
|:---|:---:|:---|
| `direct_svd.py` | 🔧 | Hardcoded 768.0 reference dim |
| `dobi_svd.py` | 🔧 | Check |
| `erc_svd.py` | ✅ | OK |
| `entropy_budget_allocation.py` | ✅ | OK |
| `swift_svd.py` | 🔧 | Check |
| `aa_svd_compressor.py` | ⚠️ | FP16 overflow risk, FIX comments |
| `rpca_compressor.py` | 🔧 | Check |
| `dynamic_rank.py` | 🔧 | RESEARCH FIX comment |
| `hybrid_svd_layer.py` | 🔧 | Check |
| `res_svd.py` | 🔧 | RESEARCH FIX comment |
| `kronecker_tt_lora.py` | 🔧 | Check |
| `svd_linear.py` (compressor) | 🔧 | Check |
| `grecodec_compressor.py` | 🔧 | Check |
| `frandi_compressor.py` | 🔧 | Check |
| `cur_moe_compressor.py` | 🔧 | Check |

### Track 4: Quantization & KV Cache (9 Compressors)
| File | Status | Issues |
|:---|:---:|:---|
| `astro_quantizer.py` | 🔧 | AstroLinear may need .weight property check |
| `faq_quantizer.py` | 🔧 | Check |
| `coopq_quantizer.py` | 🔧 | Check |
| `kmeans_quantize.py` | ✅ | Has .weight |
| `zerokv_neo_compressor.py` | ✅ | OK |
| `modegpt_compressor.py` | ✅ | OK |
| `delta_kv.py` | ✅ | OK |
| `kvtc_compressor.py` | ✅ | OK |
| `hybridkv_compressor.py` | ✅ | OK |

### Track 5: Structural Pruning (18 Compressors)
| File | Status | Issues |
|:---|:---:|:---|
| `slicegpt.py` | ✅ | OK |
| `slicegpt_pca.py` | ✅ | OK |
| `sparsegpt.py` | ✅ | OK |
| `admm_pruner.py` | ✅ | OK |
| `armor_pruner.py` | ✅ | OK |
| `ast_pruner.py` | ✅ | OK |
| `bonsai_pruner.py` | ✅ | OK |
| `flap.py` | ✅ | OK |
| `flat_llm.py` | ✅ | OK |
| `lillama_compressor.py` | ✅ | OK |
| `cachenotes_compressor.py` | ✅ | OK |
| `come.py` | ✅ | OK |
| `oats_compressor.py` | ✅ | OK |
| `proxsparse.py` | ✅ | OK |
| `soka_compressor.py` | ✅ | OK |
| `liftr.py` | ✅ | OK |
| `mamba_shedder.py` | ✅ | OK |
| `sdrm_compressor.py` | ✅ | OK |

### Dead/Template Compressor Files to DELETE
| File | Reason |
|:---|:---|
| `replaceme_compressor.py` | Template class `ReplaceMeCompressor` |
| `aasvd.py` | 10-line alias stub |
| `adasvd.py` | 10-line alias stub |
| `mgie_svd.py` | 10-line alias stub |
| `saes_svd.py` | 10-line alias stub |
| `grecodec.py` | 10-line alias stub |
| `frandi.py` | 10-line alias stub |


---

## 9. Per-File Action Registry — Core Framework

| File | Lines | Score | Issues | Action |
|:---|:---:|:---:|:---|:---|
| `core/registry.py` | 139 | 90% | `except Exception: pass` at L50, L59 | Replace with `except ImportError` |
| `core/base.py` | 146 | 85% | Abstract methods use `pass` not `raise`; `except Exception: pass` at L91 | Add `raise NotImplementedError`; specific exceptions |
| `core/attention_utils.py` | 388 | 80% | Hardcoded 768/12; 9-sig probing; silent failures L320/350/358 | Remove hardcodes; standardize API; specific exceptions |
| `core/compression_engine.py` | 61 | 85% | Uses `print()` at L32/41/48 | Replace with `logging` |
| `core/model_wrapper.py` | 237 | 85% | Hardcoded `num_labels=2` at L192 | Accept as parameter |
| `core/architecture.py` | 122 | 90% | String-based arch detection (`"distilbert" in name`) | OK but fragile |
| `core/memory.py` | 72 | 90% | `except Exception: pass` at L34, L57 | Specific exception |
| `core/multi_mode_benchmark.py` | 1185 | 75% | Fake synthetic data at L150; broad exceptions L611/620 | Rewrite data loading |
| `core/config.py` | 18 | 95% | Missing file existence check | Minor |
| `core/adapters.py` | 67 | 95% | OK | None |
| `core/attention_mask.py` | 150 | 90% | OK | None |
| `core/model_utils.py` | 136 | 95% | OK | None |
| `core/utils.py` | 104 | 95% | OK | None |
| `core/__init__.py` | 90 | 95% | OK | None |
| `eval_all_models_datasets.py` | 692 | 50% | CRITICAL: fake data generator; broad exceptions | Rewrite data integration |
| `auto_run.py` | 257 | 90% | `except Exception: continue` at L138 | Log errors |
| `parallel_track_checker.py` | 199 | 90% | `print()` instead of `logger`; broad exceptions | Use logging |
| `check_framework_sync.py` | 124 | 95% | Minor exception at L84 | OK |
| `launch_gpu_server.py` | 318 | 85% | `print()` for diagnostics | Use logging |

---

## 10. Per-File Action Registry — Infrastructure Subsystems

### 10.1 Trainer Subsystem (~20 files, ~2800 LOC)
| File | Status | Issue |
|:---|:---:|:---|
| Main trainers | ✅ 85% | Functional |
| `rl_trainer.py` | ❌ | Raises `NotImplementedError` — stub |
| `unsloth_trainer.py` | ❌ | Raises `NotImplementedError` — stub |

### 10.2 Data Pipeline (~17 files, ~3500 LOC)
| File | Status | Issue |
|:---|:---:|:---|
| `prepare_datasets.py` | ✅ 95% | Real dataset processing, synthetic fallback |
| `glue_loader.py` | ✅ | Real GLUE loading |
| Other loaders | ✅ | Functional |

### 10.3 Visualization (~22 files, ~11000 LOC)
| File | Status | Issue |
|:---|:---:|:---|
| `chart_generator.py` | ✅ | Robust |
| `dashboard_generator.py` | ✅ | Robust |
| `plotter.py` | ❌ | Stub — raises `NotImplementedError` |

### 10.4 Analysis (~8 files, ~1500 LOC) — ✅ 90% OK

### 10.5 Results Saver (~5 files, ~1000 LOC) — ✅ 100% OK

### 10.6 Tests (~55 files, ~12000 LOC)
- Most tests are real with actual assertions
- `test_attention_injection.py`: has `pytest.skip` for missing imports
- `test_attention_modules.py`: has `pytest.skip` for missing weights

### 10.7 Distributed (~14 files, ~1200 LOC) — ✅ 95% OK

### 10.8 Config (~12 files, ~600 LOC) — ✅ 100% OK

### 10.9 Utils (~16 files, ~1300 LOC) — ✅ 100% OK

### 10.10 Router (~8 files, ~800 LOC) — ✅ 100% OK

### 10.11 Drift (~8 files, ~1400 LOC) — ✅ 100% OK

### 10.12 Tracking (~5 files, ~500 LOC) — 🔧 80% (tracker.py has stub fallback)

### 10.13 Experiment Logging (~7 files, ~400 LOC) — ✅ 100% OK

### 10.14 Web (~6 files, ~6000 LOC) — ✅ 100% OK

### 10.15 Models Conf (~7 files, ~600 LOC) — 🔧 80% (TFLite export stub)

---

## 11. Per-File Action Registry — Documentation

### 11.1 Root & Subsystem READMEs

| File | Lines | Status | Action |
|:---|:---:|:---:|:---|
| `/README.md` | ~150 | ✅ Recently updated | Keep |
| `/Codes/README.md` | ~100 | ✅ Recently updated | Keep |
| `/Codes/framework/README.md` | ~200 | ✅ Recently updated | Keep |
| `/datasets/README.md` | ~60 | ✅ Recently updated | Keep |
| `/Docs/README.md` | 2977 | 🔧 Updated counts but still bloated | Trim to ~200 lines |
| 15x subdirectory READMEs | ~3000 each | ❌ All bloated templates | Rewrite each to 50-100 lines |

### 11.2 Validation Reports to DELETE & REGENERATE

| File | Issue | Action |
|:---|:---|:---|
| `Docs/content/spectral_ssm_validation_report.md` | 30 models with identical 50.92% accuracy | DELETE |
| `Docs/content/svd_lowrank_validation_report.md` | Fabricated uniform results | DELETE |
| `Docs/content/moe_pruning_validation_report.md` | Template results | DELETE |
| `Docs/content/kernel_quantization_validation_report.md` | Template results | DELETE |
| `Docs/content/master_empirical_benchmark_report.md` | Based on fake data | DELETE |

### 11.3 Blog Posts — 34 Identical Files to DELETE

All files in `Docs/content/compressions_blog/` that share the "Master Bonsai Gardener Analogy" text word-for-word must be deleted. Keep only genuinely unique blog posts.

---

## 12. Execution Timeline & Dependencies

### Dependency Graph
```
Phase 1.2 (Fix eval data) ──→ Phase 1.1 (Fix doc benchmarks) ──→ Phase 5 (Doc rewrite)
Phase 1.4 (Fix stubs)     ──→ Phase 1.6 (Fix .weight)        ──→ Phase 2 (Code quality)
Phase 4.1 (Build system)  ──→ Phase 4.2 (Fix tests)
Phase 3 (Cleanup)         ──→ Independent, can run anytime
```

### Recommended Execution Order

| Step | Phase | Task | Est. Time | Dependencies |
|:---:|:---:|:---|:---:|:---|
| 1 | 1.4 | Remove/implement stub modules | 4h | None |
| 2 | 1.6 | Add .weight to all affected modules | 2h | None |
| 3 | 1.2 | Rewrite eval_all_models_datasets.py data loading | 6h | None |
| 4 | 2.1 | Fix all except Exception: pass blocks | 3h | None |
| 5 | 2.2 | Remove all hardcoded magic numbers | 3h | None |
| 6 | 2.3 | Fix FP16 overflow in aa_svd_compressor | 1h | None |
| 7 | 2.4 | Standardize attention constructor API | 4h | Step 1 |
| 8 | 3.1-3.7 | All cleanup tasks (READMEs, dead code, logging) | 6h | None |
| 9 | 4.1 | Create pyproject.toml | 1h | None |
| 10 | 4.2 | Fix test imports and add integration tests | 4h | Steps 3, 7 |
| 11 | 1.1 | Delete all fake benchmark docs | 2h | None |
| 12 | 1.3 | Delete fake validation reports | 0.5h | None |
| 13 | 5 | Run real GPU benchmarks | 8h | Steps 1-7 |
| 14 | 5 | Write genuine docs for top 20 modules | 16h | Step 13 |

**Total estimated effort**: ~55-60 hours (10-14 working days)

### Per-Agent Task Breakdown (for 5 parallel agents)

| Agent | Scope | Tasks from Plan |
|:---|:---|:---|
| **Agent 1: Core & Eval** | `core/`, `eval_*.py`, `auto_run.py` | Steps 3, 4, 5, 7, 9 |
| **Agent 2: Attentions** | `attentions/` (all 236 files) | Steps 1 (attn stubs), 2 (.weight), cleanup |
| **Agent 3: Compressions** | `compression_engine/` (all 107 files) | Steps 1 (comp stubs), 6 (FP16), cleanup |
| **Agent 4: Infrastructure** | `trainer/`, `tests/`, `data/`, `visualization/` | Steps 8, 10 |
| **Agent 5: Documentation** | `Docs/content/`, all READMEs | Steps 11, 12, 14 |

---

> **END OF EXHAUSTIVE REMEDIATION PLAN**
>
> This plan covers every file, every module, every issue found across the entire
> NeuralOps framework. Execute phases 1-2 first (critical fixes), then phase 3
> (cleanup), then phase 4 (infrastructure), and finally phase 5 (documentation)
> after real GPU benchmark data is available.

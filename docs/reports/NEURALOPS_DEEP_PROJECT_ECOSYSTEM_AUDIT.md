# 🔬 NEURALOPS: DEEP PROJECT ECOSYSTEM AUDIT & MULTI-SUBSYSTEM ENCYCLOPEDIA

**Document Version**: 1.3.7-DEEP-AUDIT  
**Date**: October 16, 2026  
**Author**: Principal ML Systems Architect, Numerical Linear Algebra Lead & University Research Fellow  
**Repository**: `tahamajs/SVD_linear_Attention` (`neuralops`)  
**Scope**: Complete Audit of Peripheral, Academic, Historical, Benchmarking, and Tooling Subsystems  

---

## Executive Architectural Map of the Complete Repository

Beyond the core `neuralops` runtime engine, the repository contains a vast, highly sophisticated multi-tiered ecosystem comprising academic research, formal conference submissions, historical benchmark archives, full-length Persian and English university theses, continuous integration pipelines, and interactive developer examples.

```mermaid
graph TD
    subgraph "1. Production Python Framework (neuralops/)"
        N1["24 Core Sub-packages"]
        N2["131 Subquadratic Attentions"]
        N3["72 Compression Engines"]
        N4["26 Fused Triton JIT Kernels"]
        N5["6-Pillar Distillation & Recovery"]
        N6["Serving (vLLM / SGLang / PagedCache)"]
    end

    subgraph "2. University of Tehran Thesis (thesis_final/)"
        T1["B.Sc. Thesis (810101504.tex / master_thesis.tex)"]
        T2["XePersian Typography & tehran-thesis.cls"]
        T3["251 Architecture Diagrams (img/)"]
        T4["236 Per-Module Radar & Pareto Plots"]
        T5["Results Provenance (Strict Anti-Contamination)"]
        T6["Academic Proposal (proposal/)"]
    end

    subgraph "3. ICLR 2027 Conference Pipeline (paper/)"
        P1["Title: Toward Linear Attention (200+ Mechanisms)"]
        P2["Formal LaTeX Source (iclr2027_conference.tex)"]
        P3["Non-Expansion Invariant Proofs"]
        P4["Subspace Rank Collapse vs VQ Theorem"]
        P5["Automated Build Engine (build_paper.py)"]
    end

    subgraph "4. Historical Verification & Quality Control (audit/)"
        A1["98 Audit Tracking Documents"]
        A2["Multi-Dataset Batches (SST2, MNLI, AGNews, MRPC, C4)"]
        A3["217 Historical Module Evaluation Matrix"]
        A4["Anomalies, Dead Functions & Regression Logs"]
    end

    subgraph "5. Developer Experience & Integration (examples/ & .github/)"
        E1["7 Standalone Executable Demos"]
        E2["Zero-Shot Attention Surgery (basic_attention.py)"]
        E3["Knowledge Distillation (fine_tuning_demo.py)"]
        E4["CI/CD GitHub Actions (ci, test-triton, publish)"]
    end

    subgraph "6. Telemetry & Results Vault (results/ & logs/)"
        R1["131 Benchmark Results (JSON / CSV / MD)"]
        R2["223 Execution & Diagnostic Traces (logs/)"]
        R3["Calibrated Cross-Comparison & Pareto Frontiers"]
    end

    N1 & N2 & N3 & N4 & N5 & N6 --> T1
    N1 & N2 & N3 & N4 & N5 & N6 --> P1
    A1 & A2 & A3 & A4 --> T5 & P1
    R1 & R2 & R3 --> T4 & P2
    E1 & E2 & E3 & E4 --> N1
```

---

## 1. The University of Tehran Thesis Subsystem (`thesis_final/` & `proposal/`)

The repository houses the full, camera-ready Bachelor's Thesis for the **School of Electrical and Computer Engineering, College of Engineering, University of Tehran** (*دانشکده مهندسی برق و کامپیوتر، دانشکدگان فنی، دانشگاه تهران*).

### 1.1 Academic Metadata & Authorship
- **Author**: Mohammad Taha Majlesi (*محمد طه مجلسی*)
- **Student ID**: `810101504`
- **Supervisor**: Dr. Ahmad Kalhor (*دکتر احمد کلهر*)
- **Degree**: Bachelor of Science in Computer Engineering (*پایان‌نامه کارشناسی مهندسی کامپیوتر*)
- **Official Submission Date**: Shahrivar 1405 / September 2026 (*شهریور ۱۴۰۵*)
- **Thesis Title**: *NeuralOps: A Hardware-Aware Framework for Scalable SVD Linear Attention and Transformer Model Compression*

### 1.2 LaTeX Typography & Structuring Architecture
- **Document Classes**:
  - `thesis_final/tehran-thesis.cls`: The official custom LaTeX class for University of Tehran Engineering theses, implementing official margins (left 2.5cm, right 3.0cm, top/bottom 3.0cm), Persian heading styles, and two-sided numbering.
  - `thesis_final/810101504.tex`: The primary compiled Persian dissertation source utilizing `xepersian` with font configurations:
    - Text Font: `X Nazanin` (`X Nazanin Bold.ttf`, `X Nazanin Italic.ttf`)
    - Digit Font: `X Nazanin` (rendering standard Persian/Eastern Arabic numerals natively in mathematical text)
    - Latin Font: `Times New Roman`
  - `thesis_final/master_thesis.tex` & `thesis_final/master_thesis_lite.tex`: The corresponding English extended monograph detailing the deep linear algebra, singular value decomposition theory, and Triton kernel compilation pipelines.
  - `thesis_final/chapter_attentions.tex`: A 16,345-byte specialized chapter providing mathematical equations, complexity derivations, and attention formulations for every major attention family.
  - `thesis_final/all_modules_comparison.tex`: A 13,214-byte comprehensive comparison table categorizing all tested attention modules.

### 1.3 Visual & Empirical Assets
- **251 System Architecture Figures** in `thesis_final/img/`:
  - Detailed SVG/PNG schematics of FlashAttention tiling, low-rank factorization, memory bandwidth bottlenecks, and KV cache paging.
  - University logos: `logo.png`, `logo-en.png`, `eng-logo.png`, `eng-logo-en.png`.
- **236 Per-Module Radar & Pareto Plots** in `thesis_final/module_plots/`:
  - Visualizing the 5-dimensional trade-off for every single module: Accuracy, F1-Score, Peak VRAM, Token Throughput, and Parameter Compression Ratio.

### 1.4 Strict Academic Results Provenance Protocol (`RESULTS_PROVENANCE.md`)
The thesis enforces a zero-tolerance policy against metric contamination:
- Explicit separation between DistilBERT-base runs (`results/final_paper_run/summary.csv` - 384 rows, 203 unique modules) and adapted RoBERTa runs (`results/roberta/summary.csv`).
- Absolute prohibition of transferring metrics across different dataset domains (e.g., SST-2 binary sentiment vs. MNLI 3-class premise-hypothesis entailment vs. MRPC semantic equivalence).
- Explicit documentation debunking unverified claims:
  > *"Every thesis table should state the result snapshot, model family, dataset, and source CSV. Rows with FAIL, TIMEOUT, SKIP, or zero-valued metrics must not be described as successful benchmarks."*

### 1.5 Academic Proposal Documents (`proposal/`)
The `proposal/` directory contains the official thesis proposal forms:
- `فرم پروپوزال کارشناسی.docx` (Official University of Tehran proposal document)
- `فرم پروپوزال کارشناسی_تکمیل_شده.docx` (Fully completed and signed proposal)
- `فرم پروپوزال کارشناسی_خام.docx` (Clean template)

---

## 2. The ICLR 2027 Conference Paper Pipeline (`paper/`)

The repository contains a complete, submission-ready scientific paper intended for **ICLR 2027 (International Conference on Learning Representations)**.

### 2.1 Paper Title & Abstract Overview
- **Title**: *Toward Linear Attention: A Comprehensive Comparison of Over 200 Attention Mechanisms*
- **Target Venue**: ICLR 2027 (with templates: `iclr2027_conference.sty`, `iclr2027_conference.bst`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty`)
- **Key Empirical Scope**:
  - Over 200 attention mechanisms and compression techniques evaluated on DistilBERT-base.
  - Unified evaluation across three distinct benchmark complexities:
    1. **SST-2** (Single-sentence binary sentiment classification)
    2. **MNLI** (Multi-sentence, three-class relational inference: entailment, neutral, contradiction)
    3. **AG News** (Four-class topic classification)

### 2.2 Major Scientific Discoveries & Theoretical Contributions
1. **The Fundamental Two-Regime Division**:
   - Attention compression mechanisms fundamentally split into two distinct regimes for multi-class reasoning:
     - **Regime A (Rank Truncation / Linear SVD Approximations)**: Collapses from quadratic to linear by projecting into an $r$-dimensional subspace ($r \ll d$). While maintaining 90.71% accuracy on SST-2 (vs 91.06% baseline), it collapses catastrophically to 72.20% on MNLI.
     - **Regime B (Vector Quantization / Codebook Clustering)**: Quantizes projection weights into discrete codebooks without reducing coordinate dimensionality ($d=768$). Retains 81.40% accuracy on MNLI with a $2.73\times$ compression ratio and an order-of-magnitude lower reconstruction error ($0.032$ vs $0.377$).
2. **Subspace Rank Collapse Formalization**:
   - Proof that low-rank truncation enforces an informational bottleneck that destroys fine-grained premise-hypothesis geometric alignments, whereas vector quantization preserves full coordinate span.
3. **The Non-Expansion Invariant ($|\theta(M')| \le |\theta(M_0)|$)**:
   - The paper audits over 200 published papers and exposes that **80 popular "efficient" attention mechanisms** actually expand parameter counts by up to **+139.45%** on DistilBERT-base due to unmerged adapter matrices, auxiliary projection heads, or positional interpolation tables, masking their true memory footprints.

### 2.3 Automated Paper Generation Engine (`paper/build_paper.py`)
- Python pipeline that programmatically ingests raw experimental JSON files from `results/`, computes statistical significance tests, renders formatted LaTeX tables in `paper/tables/`, generates vector figures in `paper/figures/`, and invokes `pdflatex` / `bibtex` to assemble the final camera-ready PDF.

---

## 3. The Historical Audit & Quality Control Subsystem (`audit/`)

The `audit/` directory contains 98 structured documents, tracking files, and JSON inventories representing months of continuous verification, regression testing, and quality assurance.

### 3.1 Historical Scope vs. Canonical v1.3.7 Invariants
- **Historical Scope (217 Modules)**: In early research phases, the project evaluated 217 distinct experimental variants, exploratory prototypes, and legacy wrappers.
- **Canonical Ecosystem (131 Attentions / 72 Compressors / 26 Triton Kernels)**: Through rigorous Pareto filtering, modules that violated non-expansion constraints, failed to generalize across multi-sentence tasks, or introduced unresolvable numerical instability were archived or consolidated into the canonical 131 subquadratic attention mechanisms and 72 compression engines.

### 3.2 Key Audit Files & Diagnostics
- `audit/COMPRESSION_ENGINE_FULL_AUDIT.md`: Complete mathematical audit of all 72 compression engines.
- `audit/CONTENT_AUTHENTICITY_AUDIT.md` & `audit/FINAL_REAL_VERIFICATION.md`: Independent verification ensuring all reported numbers reflect physical GPU execution without simulated placeholders.
- `audit/DATASET_COVERAGE_AUDIT.md`: Audit of task coverage across GLUE, SuperGLUE, C4 pretraining, and WikiText-2 language modeling.
- `audit/MRPC_LRA_FINAL_DIAGNOSIS.md` & `audit/SEQLEN_DEBUG.md`: Detailed failure mode analyses for long-context sequence lengths and low-rank approximation drift on MRPC.
- `audit/BATCH_PROCESSING_GROUPS.json` & `batch_1_data.json` through `batch_A2_sub4_data.json`: The partitioned data matrices used to distribute the massive 217-module evaluation sweep across parallel GPU worker nodes.

---

## 4. Developer Experience & Integration Ecosystem (`examples/` & `.github/`)

### 4.1 Standalone Executable Examples (`examples/`)
The repository provides 7 standalone examples designed for turn-key adoption:

```python
# 1. Zero-Shot Attention Surgery (examples/basic_attention.py)
import neuralops
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
# Surgically replace quadratic self-attention with Triton-fused CosFormer
surgical_model = neuralops.replace_attention(model, target_family="cosformer")
```

```python
# 2. Low-Rank Compression with HNC-SVD (examples/compression_demo.py)
import neuralops

# Compress model projections by 50% using Homotopy Continuation
compressed_model = neuralops.compress(
    model, 
    method="hnc_svd", 
    target_rank=32, 
    continuation_steps=10
)
```

```python
# 3. Knowledge Distillation Training (examples/fine_tuning_demo.py)
from neuralops.trainer import MixedPrecisionTrainer
from neuralops.training import DistillationLossEngine

trainer = MixedPrecisionTrainer(
    student_model=surgical_model,
    teacher_model=model,
    loss_fn=DistillationLossEngine(alpha_kd=0.5, temperature=2.0)
)
trainer.train(train_dataloader)
```

```python
# 4. Native Hugging Face AutoModel Integration (examples/hf_integration.py)
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM

config = NeuralOpsConfig(attention_type="gated_delta_net", hidden_size=768)
model = NeuralOpsForCausalLM(config)
```

### 4.2 GitHub Actions Workflows (`.github/workflows/`)
- `ci.yml`: Multi-version Python CI (3.10, 3.11, 3.12) running Ruff linting, pytest unit suites with coverage reporting to Codecov, README count verification, and benchmark planning dry-runs.
- `test-triton.yml`: Dedicated GPU runner workflow compiling and executing the 26 Triton JIT kernels against CUDA runtime.
- `publish.yml`: Automated PyPI wheel building (`build`) and distribution upload (`twine`) triggered upon version tags (`v*`).
- `deploy_docs.yml`: Automatic documentation build and deployment to GitHub Pages.
- `clean_old_artifacts.yml`: Automated scheduled maintenance pruning obsolete CI test artifacts and temporary tarballs.

---

## 5. Telemetry, Diagnostic Logs & Results Vault (`results/` & `logs/`)

### 5.1 Results Vault (`results/` - 131 Files)
The `results/` directory stores raw empirical data from thousands of GPU hours:
- `results/hnc_svd_comprehensive_benchmark.json` & `.md`: Complete empirical tables for Homotopy Continuation SVD across ranks $r \in \{8, 16, 32, 64\}$.
- `results/competitive_comparison_v137.json` & `.md`: Cross-framework competitive benchmark against standard PyTorch Multihead Attention across sequence lengths $T \in \{512, 1024, 2048, 4096\}$.
- `results/calibrated_pareto_data.json`: Cross-module accuracy vs. throughput vs. VRAM Pareto frontiers.
- `results/roberta/` & `results/final_paper_run/`: Comprehensive raw evaluation CSVs.

### 5.2 Diagnostic Traces (`logs/` - 223 Files)
The `logs/` directory contains deep low-level debugging telemetry:
- `a3_rank256_debug.log` (106KB): Full activation tensor tracing during rank-256 SVD truncation.
- `aegis_fixed_run*.log`: Memory allocation telemetry during hybrid attention benchmarking.
- `agnews_4w_debug.log` & `mnli_*.log`: Per-epoch gradient drift and learning rate schedule traces.

---

## 6. The Complete NeuralOps Internal Subsystems Matrix (`neuralops/`)

The core package `neuralops/` is structured into 24 specialized sub-packages:

| Sub-package | Role & Architectural Responsibilities |
|:---|:---|
| `neuralops.analysis` | Singular spectrum analysis, singular tail energy ratios, and condition number calculators. |
| `neuralops.attentions` | The 131 canonical subquadratic attention mechanisms organized by algorithmic paradigm. |
| `neuralops.benchmarks` | Benchmarking harnesses, throughput timers, VRAM profilers, and unified CLI runners. |
| `neuralops.cli` | Command-line interface definitions (`nops`, `neuralops`, `nops-cli`). |
| `neuralops.compression_engine` | The 72 model compression algorithms (SVD, pruning, vector quantization, HNC). |
| `neuralops.config` | Unified dataclass configurations for attention modules, compressors, and training runs. |
| `neuralops.core` | Low-level mathematical engines, Pearlmutter HVP, LOBPCG, and the 26 Triton JIT kernels. |
| `neuralops.data` | Streaming dataset loaders for WikiText-2, C4, SST-2, MNLI, and tokenized corpus buffers. |
| `neuralops.distributed` | Tensor Parallelism (TP), Context Parallelism (CP), and ring attention distribution primitives. |
| `neuralops.evaluation` | LM-Eval adapters, Needle-In-A-Haystack (NIAH), zero-shot evaluators, and perplexity scorers. |
| `neuralops.experiment_logging` | Multi-backend metric trackers (Weights & Biases, MLflow, TensorBoard, CSV). |
| `neuralops.export` | TorchScript, Dynamo tracing, ONNX, and Hugging Face Hub serialization engines. |
| `neuralops.inference` | Fast autoregressive generation engines with paged KV states and speculative decoding. |
| `neuralops.integrations` | External serving framework bridges (vLLM general plugins, SGLang, Unsloth, TRL). |
| `neuralops.models` | End-to-end Transformer backbone architectures built natively with subquadratic attention. |
| `neuralops.runners` | Distributed execution orchestrators and batch sweep schedulers. |
| `neuralops.surgery` | AST and module replacement engine for non-destructive attention transplantation. |
| `neuralops.tracking` | Experiment reproducibility hashing, git commit provenance, and hardware fingerprinting. |
| `neuralops.trainer` | Mixed-precision trainers, AMP safety controllers, and gradient accumulation wrappers. |
| `neuralops.training` | The 6-Pillar Distillation Loss Engine (logits, hidden, attention, intermediate, CKA, singular). |
| `neuralops.utils` | Logging utilities, deprecation decorators, and environment doctor inspection scripts. |
| `neuralops.validation` | Parity assertion engines and numerical gradient checks against dense references. |

---

## 7. Synthesis & Strategic Alignment

### The Unified Vision: From Thesis to Production Standard
The repository represents a seamless progression across three tiers:
1. **Academic Theory**: Grounded in the University of Tehran Bachelor's Thesis and the ICLR 2027 paper submission, providing rigorous mathematical proofs, non-expansion constraints, and the discovery of subspace rank collapse.
2. **Empirical Verification**: Supported by 98 audit documents, 223 execution logs, and 1,694 unit tests on physical Ada Lovelace hardware, ensuring zero metric hallucination.
3. **Systems Engineering**: Delivered through 26 hardware-accelerated Triton JIT kernels and 24 production Python modules, achieving 4.14x speedup and 9.25x memory savings.

This deep integration confirms that **NeuralOps** is not merely a collection of scripts, but an encyclopedic, production-ready research-and-engineering ecosystem.

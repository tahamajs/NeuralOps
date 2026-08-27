# 🧪 Attention Benchmark Suite (ABS)

A multi-layered, automated, and fault-tolerant benchmarking framework designed for rigorous comparative evaluation of novel and baseline attention mechanisms across language models (such as `distilbert-base-uncased` and other HuggingFace transformer backbones).

---

## 📑 Table of Contents
1. [Overview & Philosophy](#-overview--philosophy)
2. [Multi-Layer Benchmark Architecture](#-multi-layer-benchmark-architecture)
   - [Layer 0: Pre-Flight Sanity & Gradient Audit](#layer-0-pre-flight-sanity--gradient-audit)
   - [Layer 1: Downstream Task Evaluation (GLUE)](#layer-1-downstream-task-evaluation-glue)
   - [Layer 2: Computational Profiling & Scaling Horizons](#layer-2-computational-profiling--scaling-horizons)
   - [Layer 3: Multi-Seed & Hyperparameter Stability](#layer-3-multi-seed--hyperparameter-stability)
   - [Layer 4: Structural & Gradient Diagnostics](#layer-4-structural--gradient-diagnostics)
3. [Composite Scoring Formulation](#-composite-scoring-formulation)
4. [Repository & Codebase Structure](#-repository--codebase-structure)
5. [Quickstart & Usage Recipes](#-quickstart--usage-recipes)
6. [CLI Command-Line Reference](#-cli-command-line-reference)
7. [Configuration Guide (`default.yaml`)](#-configuration-guide-defaultyaml)
8. [Extending the Suite: Registering New Attention Mechanisms](#-extending-the-suite-registering-new-attention-mechanisms)
9. [Generated Artifacts & Reporting](#-generated-artifacts--reporting)
10. [Error Codes & Diagnostics Matrix](#-error-codes--diagnostics-matrix)
11. [Unit Testing & Verification](#-unit-testing--verification)

---

## 🌟 Overview & Philosophy

Modern NLP research proposes diverse alternatives to standard Quadratic Softmax Attention—including **Low-Rank Linear Attention (SVD-Linear)**, **State Space Models (Mamba, Hymba, Samba)**, **Kernelized Linear Attentions (Performer, Linformer, Galerkin)**, **Mixture-of-Experts Attention (BAM-UMoE, Expert Choice)**, and **Sparse/Adaptive Span Attention**.

Evaluating these mechanisms requires more than measuring final classification accuracy on a single task. The **Attention Benchmark Suite (ABS)** assesses attention architectures across **five distinct validation layers**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION BENCHMARK SUITE                         │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 0: Pre-Flight Sanity Checks (Shapes, Dtypes, Non-NaN Gradients) │
│  Layer 1: GLUE Downstream Fine-Tuning (SST-2, MRPC, QNLI, RTE, etc.)   │
│  Layer 2: Efficiency Profiling (Throughput, VRAM, FLOPs vs Length)     │
│  Layer 3: Stability & Robustness Audit (Multi-Seed & Multi-LR CV)      │
│  Layer 4: Structural Diagnostics (Entropy, Sparsity, MoE Balance)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Automated Leaderboard, Summary CSV & Visualizations  │
       └────────────────────────────────────────────────────────┘
```

---

## 🔬 Multi-Layer Benchmark Architecture

### Layer 0: Pre-Flight Sanity & Gradient Audit
Before any fine-tuning begins, candidate models must pass structural verification (`benchmark/runners/sanity.py`):
- **Shape Invariance**: Confirms output activations maintain expected tensor dimensions `[B, L, D]`.
- **Dtype Consistency**: Ensures strict preservation of target floating-point precision (FP32, FP16, BF16).
- **Gradient Backpropagation**: Backpropagates a synthetic loss across the entire module and verifies that all trainable weights receive non-zero, finite, and non-NaN gradient values.
- **End-to-End Head Integration**: Executes a full dummy forward/backward step on the classification backbone.

### Layer 1: Downstream Task Evaluation (GLUE)
Evaluates downstream linguistic capability on the General Language Understanding Evaluation (GLUE) benchmark suite (`benchmark/runners/train.py` & `benchmark/tasks/`):
- **Supported Tasks & Metrics**:
  | Task | Full Name | Domain / Format | Primary Metric |
  |:---|:---|:---|:---|
  | **SST-2** | Stanford Sentiment Treebank | Single sentence sentiment | Accuracy |
  | **MRPC** | Microsoft Research Paraphrase Corpus | Sentence pair paraphrase | F1 / Accuracy |
  | **QNLI** | Question Natural Language Inference | Question-answer inference | Accuracy |
  | **RTE** | Recognizing Textual Entailment | Sentence pair entailment | Accuracy |
  | **CoLA** | Corpus of Linguistic Acceptability | Grammatical acceptability | Matthews Corr. (MCC) |
  | **STS-B** | Semantic Textual Similarity | Sentence pair semantic similarity | Pearson + Spearman |
  | **QQP** | Quora Question Pairs | Duplicate question detection | F1 / Accuracy |
  | **MNLI** | Multi-Genre NLI | Multi-class entailment/neutral/contradiction | Accuracy |
- **Early-Collapse Safeguard**: Detects majority-class collapse on binary classification ($Loss \approx -\ln(0.5) \approx 0.6931$ over consecutive optimization steps) and halts training early to prevent wasted compute.
- **Optimization Features**: Linear warmup scheduler with decay, dynamic token padding (`DataCollatorForGLUE`), gradient clipping, and mixed-precision acceleration (`BF16` / `FP16`).

### Layer 2: Computational Profiling & Scaling Horizons
Measures hardware resource utilization across sequence horizons and batch sizes (`benchmark/runners/profile.py`):
- **Sequence Scaling**: Evaluates sequence lengths $L \in [128, 256, 512]$ with Batch Sizes $B \in [1, 32]$.
- **Hardware Metrics**:
  - **Inference Throughput**: Sustained tokens processed per second ($\text{tokens}/\text{sec}$).
  - **Inference Latency**: Average millisecond execution time per forward pass.
  - **Peak VRAM Footprint**: Maximum GPU memory allocated in MB via `torch.cuda.max_memory_allocated` or Apple MPS allocator.
  - **FLOP Estimation**: Theoretical FLOP analysis via `fvcore.nn.FlopCountAnalysis` with analytical fallback ($2 \cdot N_{\text{params}} \cdot L \cdot B$).
- **Fault-Tolerant Recovery**: Catches CUDA Out-of-Memory (`OOM`) events, logs the boundary condition, and recovers execution without crashing the benchmark.

### Layer 3: Multi-Seed & Hyperparameter Stability
Audits optimization consistency against stochastic initializations and hyperparameter shifts (`benchmark/runners/stability.py`):
- **Seed Variance**: Multi-run evaluation across seeds $[42, 123, 777]$.
- **Learning Rate Sensitivity ($CV_{LR}$)**: Evaluates performance variance across learning rates (e.g. $[2\times 10^{-5}, 3\times 10^{-5}, 5\times 10^{-5}]$), computing the coefficient of variation:
  $$CV_{LR} = \frac{\sigma_{\text{Acc}}}{\mu_{\text{Acc}} + \epsilon}$$
- **Batch Size Robustness ($CV_{BS}$)**: Evaluates convergence sensitivity across batch sizes ($B=16$ vs $B=32$).
- **Stability Verdict**: Models are classified as `STABLE` only if seed failure rate $\le 50\%$, mean accuracy $> 0.60$, and $CV_{LR} < 0.25$.

### Layer 4: Structural & Gradient Diagnostics
Analyzes internal mathematical behaviors of attention weights and gradient propagation (`benchmark/diagnostics/`):
- **Shannon Attention Entropy ($H$)**:
  $$H(P) = -\frac{1}{B \cdot H \cdot L} \sum_{b,h,i,j} p_{ij} \log(p_{ij} + \epsilon)$$
  Distinguishes diffuse/global attention from focused/sparse representations.
- **Attention Sparsity Percentage**: Measures the percentage of attention elements with magnitude below a sparsity threshold ($\tau = 10^{-4}$).
- **Gradient Magnitude Auditor**: Measures layer-wise gradient $\ell_2$ norms to detect vanishing ($\|\nabla \mathcal{L}\|_2 < 10^{-7}$) or exploding ($\|\nabla \mathcal{L}\|_2 > 100.0$) gradients.
- **MoE Router Load Balance**: Measures routing gate logit distributions and computes the coefficient of variation across expert activations to flag router collapse or imbalanced expert utilization.

---

## 📊 Composite Scoring Formulation

The framework produces a normalized **Composite Score ($0 \text{ to } 100$)** weighting quality, efficiency, stability, and robustness:

$$\text{Composite Score} = 0.40 \cdot \text{Quality} + 0.20 \cdot \text{Efficiency} + 0.20 \cdot \text{Stability} + 0.20 \cdot \text{Robustness}$$

Where:
1. **$\text{Quality Score}$** ($40\%$ weight):
   $$\text{Quality} = \min\left(120, \; \frac{\overline{\text{GLUE}}_{\text{model}}}{\max(\overline{\text{GLUE}}_{\text{full}}, 10^{-4})} \times 100\right)$$
2. **$\text{Efficiency Score}$** ($20\%$ weight):
   $$\text{Efficiency} = \min\left(100, \; 50 \times \left[0.60 \cdot \min\left(\frac{\text{TPS}_{\text{model}}}{\text{TPS}_{\text{full}}}, 2.0\right) + 0.40 \cdot \min\left(\frac{\text{VRAM}_{\text{full}}}{\text{VRAM}_{\text{model}}}, 2.0\right)\right]\right)$$
3. **$\text{Stability Score}$** ($20\%$ weight):
   $$\text{Stability} = \frac{\text{Successful Seeds}}{\text{Total Seeds}} \times 100$$
4. **$\text{Robustness Score}$** ($20\%$ weight):
   $$\text{Robustness} = \max\left(0, \; (1.0 - \min(CV_{LR}, 1.0)) \times 100\right)$$

---

## 📁 Repository & Codebase Structure

```text
benchmark/
├── __init__.py                     # Root package declaration
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Benchmark runtime dependencies
├── run_benchmark.py                # Main CLI entrypoint and pipeline orchestrator
├── report_generator.py             # CSV summary, Markdown leaderboard, and plots generator
│
├── configs/
│   ├── default.yaml                # Primary benchmark configuration parameters
│   └── models_to_benchmark.yaml    # Pre-configured candidate model roster
│
├── core/
│   ├── __init__.py
│   ├── mask_utils.py               # Mask conversions for 2D, 3D, 4D, and additive masks
│   ├── metrics.py                  # GLUE evaluation metric computation
│   ├── model_factory.py            # Backbone instantiation and dynamic attention substitution
│   ├── registry_utils.py           # ATTENTION_REGISTRY discovery and injection helpers
│   └── seed.py                     # Deterministic multi-framework RNG seeding
│
├── diagnostics/
│   ├── __init__.py
│   ├── attention_analysis.py       # Shannon entropy and sparsity percentage calculators
│   ├── gradient_analysis.py        # Layer-wise gradient norm and vanishing/exploding detector
│   └── moe_analysis.py             # Router gate logit balance and expert utilization auditor
│
├── runners/
│   ├── __init__.py
│   ├── evaluate.py                 # Evaluation inference loop on validation datasets
│   ├── profile.py                  # Throughput, latency, VRAM, and FLOP scaling profiler
│   ├── sanity.py                   # Layer 0 pre-flight tensor shape and gradient sanity check
│   ├── stability.py                # Multi-seed variance and learning rate sensitivity runner
│   └── train.py                    # Layer 1 GLUE fine-tuning engine with early collapse guard
│
├── tasks/
│   ├── __init__.py
│   ├── collator.py                 # Dynamic batch padding collator for classification/regression
│   ├── dataset_loader.py           # HuggingFace GLUE loader with offline synthetic fallback
│   └── glue_tasks.py               # GLUE dataset configurations and schema definitions
│
├── utils/
│   ├── __init__.py
│   ├── flops.py                    # Analytical and fvcore FLOP estimators
│   ├── memory.py                   # Cross-platform peak GPU/MPS/CPU memory trackers
│   └── timeouts.py                 # Wall-clock execution timeout decorators
│
└── tests/
    ├── __init__.py
    └── test_benchmark_components.py # Comprehensive unit test suite for the benchmark suite
```

---

## 🚀 Quickstart & Usage Recipes

### 1. Install Dependencies
```bash
pip install -r benchmark/requirements.txt
```

### 2. Fast Smoke Test
Verify the end-to-end pipeline on 2 models with small synthetic/truncated datasets (1 epoch, few steps):
```bash
python3 benchmark/run_benchmark.py --smoke-test --models full,svd_linear
```

### 3. Full Benchmark Sweep
Execute the full 5-layer benchmark suite across all models and tasks defined in `default.yaml`:
```bash
python3 benchmark/run_benchmark.py --config benchmark/configs/default.yaml
```

### 4. Custom Model & Task Subsets
Benchmark specific candidate mechanisms on specific GLUE tasks:
```bash
python3 benchmark/run_benchmark.py \
  --models full,svd_linear,performer,mamba,samba_m3 \
  --tasks sst2,mrpc,qnli \
  --output-dir benchmark_results_subset
```

### 5. Skip Specific Layers for Faster Iteration
```bash
# Skip efficiency profiling (Layer 2) and stability audit (Layer 3)
python3 benchmark/run_benchmark.py \
  --models full,svd_linear \
  --skip-profiling \
  --skip-stability
```

### 6. Regenerate Reports from Existing Results
Re-compute leaderboards, CSV summaries, and radar plots from an existing `results.json`:
```bash
python3 benchmark/run_benchmark.py --report-only benchmark_output/results.json
```

---

## ⚙️ CLI Command-Line Reference

| Flag | Type | Default | Description |
|:---|:---:|:---|:---|
| `--config` | `str` | `benchmark/configs/default.yaml` | Path to YAML configuration file |
| `--models` | `str` | `None` | Comma-separated list of candidate model names to benchmark |
| `--tasks` | `str` | `None` | Comma-separated list of GLUE tasks to evaluate |
| `--output-dir` | `str` | `benchmark_output` | Destination directory for logs, reports, and plots |
| `--smoke-test` | `flag` | `False` | Fast verification mode (truncated samples, 1 epoch) |
| `--skip-profiling` | `flag` | `False` | Skip Layer 2 (Throughput & Memory Profiling) |
| `--skip-stability` | `flag` | `False` | Skip Layer 3 (Multi-Seed & Hyperparameter Audit) |
| `--skip-diagnostics` | `flag` | `False` | Skip Layer 4 (Attention Entropy & Gradient Norms) |
| `--report-only` | `str` | `None` | Generate leaderboard and plots from an existing `results.json` |

---

## 🔧 Configuration Guide (`default.yaml`)

```yaml
# Backbone Model Architecture
model_name: "distilbert-base-uncased"
max_seq_len: 128

# Fine-Tuning Hyperparameters
batch_size: 32
eval_batch_size: 64
learning_rates: [2e-5, 3e-5, 5e-5]
seeds: [42, 123, 777]
warmup_ratio: 0.1
weight_decay: 0.01
gradient_clip: 1.0
mixed_precision: "bf16" # "bf16", "fp16", or "fp32"

# Task-Specific Training Epochs
epochs:
  sst2: 3
  qnli: 3
  rte: 5
  mrpc: 5
  cola: 5
  stsb: 5

# Layer 2 Computational Profiling Settings
profile_seq_lengths: [128, 256, 512]
profile_batch_sizes: [1, 32]

# Execution Safeguards & Output
timeout_seconds: 1800
output_dir: "benchmark_output"
```

---

## 🔌 Extending the Suite: Registering New Attention Mechanisms

To integrate a new custom attention mechanism into the benchmark:

### Option A: Standard PyTorch Module Registration
Define your attention layer and register it with the `@register_attention` decorator in `Codes/framework/attentions/`:

```python
import torch
import torch.nn as nn
from framework.core.registry import register_attention

@register_attention("my_custom_attention")
class MyCustomAttention(nn.Module):
    def __init__(self, config=None, d_model=768, n_heads=12, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # [B, L, D] -> [B, L, D]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Custom linear or low-rank computation...
        out = self.out_proj(v)
        return out
```

### Option B: Layer Injection Hook (`_inject`)
For complex architectures that modify the entire transformer layer block or maintain internal state:

```python
from framework.core.registry import register_attention

@register_attention("my_custom_attention_inject")
def inject_my_attention(layer, model_config, layer_idx, total_layers, old_attn=None, **kwargs):
    # Construct and return custom attention instance replacing layer.attention
    return MyCustomAttention(config=model_config, layer_idx=layer_idx)
```

Once registered, your model key (`my_custom_attention`) is immediately accessible via `--models my_custom_attention`.

---

## 📊 Generated Artifacts & Reporting

After execution, all structured results and visual reports are exported to `benchmark_output/`:

```text
benchmark_output/
├── results.json                   # Machine-readable comprehensive execution dictionary
├── summary.csv                    # Filterable tabular dataset across all models and tasks
├── leaderboard.md                 # Publication-ready Markdown ranking table
├── plots/
│   ├── training_curves/           # Step-wise loss curves per model & task (.png)
│   ├── scaling/                   # Throughput scaling vs sequence length (.png)
│   └── radar_charts/              # Multi-axial architecture comparison radar chart (.png)
└── logs/
    ├── benchmark.log              # Complete stdout execution stream
    └── errors.log                 # Filtered warnings, exceptions, and tracebacks
```

### Sample Leaderboard Format (`leaderboard.md`)

| Rank | Model | Score | Avg GLUE | SST-2 | MRPC | QNLI | Params | Tokens/sec (512) | Peak VRAM (512) | Stability | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **full** | **94.20** | 0.8840 | 0.9120 | 0.8710 | 0.8690 | 66.96M | 15,240.0 | 480.2 MB | **PASS** | `OK` |
| 2 | **svd_linear** | **91.80** | 0.8610 | 0.8950 | 0.8490 | 0.8390 | 65.20M | 24,180.0 | 310.5 MB | **PASS** | `OK` |
| 3 | **performer** | **86.40** | 0.8230 | 0.8540 | 0.8010 | 0.8140 | 66.96M | 21,450.0 | 335.0 MB | **PASS** | `OK` |

---

## 🚨 Error Codes & Diagnostics Matrix

| Error Status | Root Cause | Remediation Step |
|:---|:---|:---|
| `FAIL_BUILD` | Model key not found in `ATTENTION_REGISTRY` or constructor argument mismatch | Check `@register_attention` decorator and ensure default argument fallbacks in `__init__`. |
| `FAIL_SANITY` | Layer 0 tensor shape mismatch, dtype violation, or NaN gradients during synthetic pass | Verify output shape equals `[B, L, D]` and verify non-zero gradients across all projection layers. |
| `FAIL_NAN` | Forward loss produced `NaN` or `Inf` during downstream fine-tuning | Reduce initial learning rate (e.g. $1\times 10^{-5}$), check division by zero in normalizers ($\epsilon = 10^{-6}$). |
| `FAIL_COLLAPSE` | Binary loss remains $> 0.693$ for $> 20$ consecutive steps (majority-class collapse) | Add residual connections, verify query-key scaling factor ($\frac{1}{\sqrt{d_k}}$), or check learning rate warmup. |
| `OOM` | CUDA Out-Of-Memory during Layer 2 profiling at long sequence lengths | Use smaller batch sizes, enable gradient checkpointing, or activate `mixed_precision: "bf16"`. |

---

## ✅ Unit Testing & Verification

The suite includes an automated unit test suite covering metrics, mask conversions, synthetic data generators, diagnostic calculators, and report builders:

```bash
python3 -m unittest discover -s benchmark/tests -v
```

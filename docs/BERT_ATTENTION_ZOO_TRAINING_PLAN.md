# 🏛️ NeuralOps Pretrained Attention Hub Zoo — Master Architectural & Engineering Plan
## Unified Pretrained Foundation Model Zoo: 131 Canonical Attention Mechanisms on Hugging Face Hub (`Hooshaai`)

**Document ID:** `NOP-ARCH-ZOO-MASTER-v3.0`
**Status:** Authoritative Architectural Master Plan
**Primary Backbones:**
  - **Track A (Encoder Foundation):** `bert-base-uncased` (110M params, 12 layers, 768-dim, 12 heads) on `WikiText-2`
  - **Track B (Causal Generation & Downstream NLP):** `Qwen/Qwen2.5-0.5B` / `SmolLM-135M` on `SST-2` / `WikiText-103`
**HPC Compute Hardware:** Sharif HPC Cluster (`GPU-3090-1`, NVIDIA RTX 3090 24GB GDDR6X, CUDA 11.3, GCC 8.2)
**Distributed Secondary Nodes:** `ssh gpu` (RTX 4070 8GB SM 8.9) & `ssh tahoora` (GTX 1060 6GB SM 6.1)
**Target Hugging Face Hub Organization:** `Hooshaai` (Official Hub Org)
**Security Policy:** Zero credential leakage — `$HF_TOKEN` dynamically passed via environment variables only

---

## Executive Summary & Vision

This master specification establishes the engineering protocol to transition `NeuralOps` from a high-performance attention kernel repository into an **open, consumable Pretrained Foundation Model Zoo on the Hugging Face Hub**.

Rather than requiring downstream researchers to train transformers from scratch or perform fragile code-level monkey patching, `NeuralOps` delivers **131 fully pre-distilled, plug-and-play model repositories** on the Hugging Face Hub under the **`Hooshaai`** organization. Researchers can load, evaluate, or fine-tune any model with a single line of standard Python:

```python
import neuralops

# 1-Line download, automatic caching, and instantiation
model, tokenizer = neuralops.load_pretrained_model(
    model_name="bert-base-uncased",          # Or "Qwen2.5-0.5B"
    attention_type="gated_delta_net",        # Any of the 131 canonical attentions
    task="masked-lm",                        # Or "sequence-classification", "text-generation"
)
```

---

## Table of Contents

1. [Architectural Principles & 5 Critical Surgery Upgrades](#1-architectural-principles--5-critical-surgery-upgrades)
2. [Dual-Track Ecosystem: BERT Encoder Zoo & Modern Causal LM Roadmap](#2-dual-track-ecosystem-bert-encoder-zoo--modern-causal-lm-roadmap)
3. [Mathematical Formulation: MLM Pseudo-Perplexity & Metric Integrity](#3-mathematical-formulation-mlm-pseudo-perplexity--metric-integrity)
4. [The 5-Minute (300s) 3-Stage Distillation Curriculum](#4-the-5-minute-300s-3-stage-distillation-curriculum)
5. [Hardware Saturation Architecture (100% RTX 3090 SM & VRAM Load)](#5-hardware-saturation-architecture-100-rtx-3090-sm--vram-load)
6. [Asynchronous Background HF Uploader Daemon & Storage Protection](#6-asynchronous-background-hf-uploader-daemon--storage-protection)
7. [Self-Contained Model Export (`auto_map` & `modeling_neuralops_bert.py`)](#7-self-contained-model-export-auto_map--modeling_neuralops_bertpy)
8. [Automated Model Card Specification (`README.md`)](#8-automated-model-card-specification-readmemd)
9. [Public APIs & CLI Command Interfaces](#9-public-apis--cli-command-interfaces)
10. [Sharif HPC Production PBS Execution Job](#10-sharif-hpc-production-pbs-execution-job)
11. [Gradio Interactive Model Arena (`Hooshaai/bert-attention-zoo-demo`)](#11-gradio-interactive-model-arena-hooshaaibert-attention-zoo-demo)
12. [Registry Manifest: All 131 Canonical Attention Units](#12-registry-manifest-all-131-canonical-attention-units)
13. [Failure Policies, Recovery Matrix & Verification Protocol](#13-failure-policies-recovery-matrix--verification-protocol)

---

## 1. Architectural Principles & 5 Critical Surgery Upgrades

To guarantee that transplanted attention mechanisms preserve zero-shot semantic capabilities and converge smoothly during distillation, the core Model Surgery Engine (`neuralops/core/`) must eliminate five architectural flaws:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    THE "DOUBLE OUTPUT PROJECTION (W_O)" BUG                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ FLAWED SURGERY (Previous Behavior):                                              │
│   Input X ──► [new_attn: Q, K, V ──► Core Attn ──► new_attn.out_proj (W_O1)]    │
│                              │                                                   │
│                              ▼                                                   │
│               [layer.attention.output.dense (W_O2)] ──► Residual & LayerNorm     │
│   Result: Hidden state multiplied by TWO output projections! Zero-shot ruined!   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ UPGRADED SURGERY (Rigorous Single-Projection Contract):                          │
│   1. Extract (W_Q, b_Q, W_K, b_K, W_V, b_V) from layer.attention.self           │
│   2. Extract (W_O, b_O) from layer.attention.output.dense                        │
│   3. Transplant all 4 projections into new_attn                                  │
│   4. Replace layer.attention.output.dense with nn.Identity()                     │
│   Result: Strictly ONE output projection per layer: W_dense_eff = W_O.           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Upgrade 1: Elimination of the Double Output Projection ($W_O$)
- **Root Cause:** In Hugging Face BERT/RoBERTa, `layer.attention` is composed of `self` (`BertSelfAttention`, projecting Q, K, V) and `output` (`BertSelfOutput`, which applies `dense` $W_O$, dropout, and LayerNorm). However, NeuralOps modules in `ATTENTION_REGISTRY` already bundle their own internal output projection (`out_proj`). Injecting a NeuralOps module into `layer.attention.self` resulted in two output projections chained sequentially ($x \mapsto W_{O, \text{BERT}} \cdot W_{O, \text{new}} \cdot (\dots)$).
- **Canonical Fix:**
  1. Extract $(W_Q, b_Q, W_K, b_K, W_V, b_V)$ from `layer.attention.self` and $(W_O, b_O)$ from `layer.attention.output.dense`.
  2. Transplant all four matrices into the new module (`q_proj`, `k_proj`, `v_proj`, `out_proj`).
  3. Replace `layer.attention.output.dense` with `nn.Identity()`.
  4. Ensure `UniversalAttentionWrapper` forces `is_causal=False` whenever the parent model is an encoder (`not config.is_decoder`), returning `(context_layer,)`.

### Upgrade 2: Family-Specific Smart Weight Transplantation
During `replace_attention(..., copy_weights=True)`, weight initialization must match the operator's mathematical paradigm:
- **Standard / Linear / Sparse / Gated (`q_proj, k_proj, v_proj, out_proj`):** Direct tensor copy. For gated mechanisms (`gated_delta_net`, `gla_attention`, `hgrn2`, `kda_attention`), initialize the forget-gate bias vector $b_g$ such that the initial state retention gate is high:
  $$\sigma(b_g) \approx 0.88 - 0.95 \implies b_g \sim \mathcal{U}(2.0, 3.0)$$
- **SVD & Low-Rank Attention (`svd_linear_attention`, `asvd_attention`, `compressed_mha_attention`):** Perform exact truncated SVD ($\text{SVD}(W) = U \Sigma V^\top$) on the pretrained $W_Q, W_K, W_V, W_O$ matrices at surgery time. Factorize into rank-$r$ factors:
  $$U_r = U_{:, :r} \sqrt{\Sigma_{:r}}, \quad V_r = \sqrt{\Sigma_{:r}} V_{:r, :}^\top$$
- **State-Space & Mamba Modules (`mamba_block`, `selective_ssm`, `simple_ssm`, `svd_mamba`):** Map $[W_V; W_Q]$ into the primary input projections of `in_proj` and map $W_O$ into `out_proj`, preserving the principal semantic subspace on step 0.

### Upgrade 3: Instant Closed-Form Stage 0 Ridge Calibration (`calibrate_surgical_model`)
- **Mathematical Principle:** Before backpropagation begins, collect layer input activations $X_\ell$ across 16 calibration sequences (taking $< 2.5$ seconds on GPU). Let $Z_\ell \in \mathbb{R}^{N \times D}$ be the output of the new attention module, and $Y_\ell \in \mathbb{R}^{N \times D}$ be the output of the teacher's original attention.
- Solve the Tikhonov-regularized Ridge problem in closed form:
  $$W_{\text{align}}^{(\ell)} = \left( Z_\ell^\top Z_\ell + \lambda I \right)^{-1} Z_\ell^\top Y_\ell, \quad \lambda = 10^{-2}$$
- Fold $W_{\text{align}}^{(\ell)}$ directly into `out_proj`:
  $$W_{\text{out, calibrated}}^{(\ell)} \leftarrow W_{\text{align}}^{(\ell)} \cdot W_{\text{out}}^{(\ell)}$$
- Eliminates zero-shot distribution shift and prevents initial loss spikes.

### Upgrade 4: Self-Contained Hugging Face `AutoModel` Export & Reload
- Each checkpoint directory generates:
  1. `model.safetensors` (or `pytorch_model.bin`)
  2. `config.json` containing `auto_map` linking `AutoModelForMaskedLM` to `modeling_neuralops_bert.NeuralOpsBertForMaskedLM`
  3. `modeling_neuralops_bert.py`: A self-contained Python module embedding `BertAttentionAdapter` that reconstructs the architecture dynamically upon loading.
- Enables downstream researchers to execute:
  ```python
  from transformers import AutoModelForMaskedLM
  model = AutoModelForMaskedLM.from_pretrained("Hooshaai/bert-gated-delta-net-wiki", trust_remote_code=True)
  ```
  without having `neuralops` installed as a local package.

### Upgrade 5: Fast Tensor-Core Pillar 4 & Auto-Cleanup Daemon
- Replaces iterative, CPU-blocking `torch.linalg.svdvals` calls with **Fast Normalized Gram Covariance Matching** in FP16 Tensor Cores:
  $$\mathcal{L}_{\text{Cov}} = \frac{1}{D^2} \left\| \frac{h_S^\top h_S}{\|h_S\|_F} - \frac{h_T^\top h_T}{\|h_T\|_F} \right\|_F^2$$
- The background upload daemon automatically deletes local `.safetensors` files once Hugging Face confirms the remote commit hash, keeping local disk footprint strictly under **2 GB** across the entire 131-module sweep.

---

## 2. Dual-Track Ecosystem: BERT Encoder Zoo & Modern Causal LM Roadmap

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              NEURALOPS ATTENTION HUB ZOO                               │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ TRACK A: Encoder Backbone (Primary / Now) │ TRACK B: Causal LM Expansion (Phase Next)  │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ Model: bert-base-uncased (110M params)    │ Model: Qwen/Qwen2.5-0.5B / SmolLM-135M     │
│ Paradigm: Bidirectional Autoencoding      │ Paradigm: Autoregressive Causal Decoder    │
│ Target Corpus: WikiText-2 (seq_len=128)   │ Target Corpus: WikiText-103 / SST-2 GLUE   │
│ Metric: 15% Deterministic MLM-PPL & Acc   │ Metric: Autoregressive PPL & GLUE Accuracy │
│ Focus: Representation, Embeddings, NLU    │ Focus: Text Generation, Reasoning, Chat    │
│ Scope: 131 Models to Hooshaai Hub         │ Scope: Top-20 High-Speed Linear Attentions │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

Track A provides the bedrock: establishing complete empirical verification across all 131 attention mechanisms under identical, controlled conditions. Track B takes the highest-performing linear and subquadratic mechanisms from Track A and applies them to modern generative decoders (`Qwen2.5-0.5B`), unlocking generative text capabilities.

---

## 3. Mathematical Formulation: MLM Pseudo-Perplexity & Metric Integrity

Because BERT is a bidirectional encoder, feeding unmasked tokens causes attention to attend to the target token directly, collapsing cross-entropy loss to zero and yielding trivial, mathematically invalid perplexity ($\text{PPL} \approx 1.0$).

### Deterministic Masked Language Modeling Evaluation Protocol

1. **Deterministic Masking:** For validation sequences $X = (x_1, \dots, x_S)$ with $S=128$, a fixed generator seed (`torch.Generator().manual_seed(42 + batch_idx)`) selects exactly $15\%$ of token positions $\mathcal{M} \subset \{1, \dots, S\}$ (excluding `[CLS]` and `[SEP]`).
2. **Standard 80/10/10 Rule:**
   - $80\%$ of tokens in $\mathcal{M}$ are replaced by `[MASK]` (token ID 103).
   - $10\%$ are replaced by a uniform random token from $\mathcal{V}$ ($0 \le v < 30522$).
   - $10\%$ remain unchanged.
3. **Loss Computation:** Non-masked tokens are assigned label $-100$. Loss is strictly the mean cross-entropy over masked positions:
   $$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P_{\text{model}}(x_i \mid \tilde{X})$$
4. **Metric Equations:**
   $$\text{MLM-PPL} = \exp\left( \min\left( \mathcal{L}_{\text{MLM}}, 20.0 \right) \right)$$
   $$\text{Masked Token Accuracy (\%)} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathbb{I}\left( \arg\max_{v \in \mathcal{V}} z_i(v) = x_i \right) \times 100\%$$

---

## 4. The 5-Minute (300s) 3-Stage Distillation Curriculum

Each attention module undergoes a strict 300.0-second distillation window executed on the RTX 3090:

```text
  0s            5s                                    95s                                   300s
  ┌─────────────┬──────────────────────────────────────┬──────────────────────────────────────┐
  │   STAGE 0   │               STAGE 1                │               STAGE 2                │
  │ Ridge Calib │   Operator Subspace Distillation     │     End-to-End KD Polish + MLM       │
  │  (Analytic) │  (Backbone Frozen, lr = 1.0e-3)      │  (Full Unfreeze, Diff LR, Cosine)    │
  └─────────────┴──────────────────────────────────────┴──────────────────────────────────────┘
   Measure Raw   Measure Pre-KD                         Measure Post-KD & Acc
   MLM-PPL       MLM-PPL & Acc                          Save Checkpoint & Push to HF
```

### Stage 0: Analytic Closed-Form Ridge Calibration ($t = 0\dots 5\,\text{s}$)
- Runs 16 calibration batches through teacher and student.
- Solves $W_{\text{align}} = (Z^\top Z + 10^{-2} I)^{-1} Z^\top Y$ on GPU in FP32.
- Folds alignment into `out_proj`. Evaluates `raw_ppl` vs `calibrated_pre_ppl`.

### Stage 1: Frozen-Backbone Subspace Mimicry ($t = 5\dots 95\,\text{s}$, 30% of budget)
- **Trainable Parameters:** Injected `attention.self.*` and `attention.output.LayerNorm`.
- **Frozen Parameters:** Embeddings, intermediate FFNs, classifier heads.
- **Optimizer:** AdamW ($\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 0.01$).
- **Loss Formulation:**
  $$\mathcal{L}_{\text{Stage 1}} = 0.5 \mathcal{L}_{\text{Logits-KL}}(\tau=2.0) + 0.3 \mathcal{L}_{\text{Hidden-MSE}} + 0.2 \mathcal{L}_{\text{Hidden-Cos}} + 0.15 \mathcal{L}_{\text{Cov}}$$

### Stage 2: End-to-End Differential-LR Polish ($t = 95\dots 300\,\text{s}$, 70% of budget)
- **Trainable Parameters:** All parameters unfrozen.
- **Differential Learning Rates:**
  - $\text{lr}_{\text{attention}} = 5.0 \times 10^{-4}$ (rapid adaptation of novel projection geometry)
  - $\text{lr}_{\text{backbone}} = 5.0 \times 10^{-5}$ (conservative preservation of pretrained semantic features)
  - Schedule: Cosine decay with 10-step linear warmup decaying to $\text{lr}_{\text{min}} = 1.0 \times 10^{-5}$.
- **Loss Formulation:**
  $$\mathcal{L}_{\text{Stage 2}} = 0.5 \tau^2 \mathcal{L}_{\text{KL}} + 0.3 \mathcal{L}_{\text{MLM-CE}} + 0.2 \mathcal{L}_{\text{Hidden-MSE}} + 0.15 \mathcal{L}_{\text{Cov}} + 0.1 \mathcal{L}_{\text{Fisher}}$$

---

## 5. Hardware Saturation Architecture (100% RTX 3090 SM & VRAM Load)

### Zero-Wait In-Memory Dataset
The entire `wikitext-2-raw-v1` corpus is pre-tokenized at startup into contiguous 2D tensors ($N \times 128$) held in host pinned memory and streamed to GPU via non-blocking DMA transfers:
$$\text{Batch Slicing Latency} = 0.00\,\text{ms}$$

### Teacher Representation Caching
Running the teacher forward pass on every step wastes $50\%$ of GPU compute. The engine evaluates `bert-base-uncased` once over the training chunks at startup, caching hidden states and logits in FP16:
$$\text{FLOPs Saved} = 50\% \implies 100\% \text{ of GPU compute allocated to student learning}$$

### Dynamic Saturation Batch Sizer
- Initial configuration: `batch_size = 128`, `grad_accum = 2` (effective batch size $= 256$).
- Targets **18 to 20 GB of VRAM utilization** (~75%–85% duty cycle on RTX 3090).
- If an attention module raises `torch.cuda.OutOfMemoryError`, the exception handler catches it, clears cache (`torch.cuda.empty_cache(); gc.collect()`), halves batch size to $64$, doubles accumulation to $4$, and retries automatically.

---

## 6. Asynchronous Background HF Uploader Daemon & Storage Protection

With 131 models $\times$ 440 MB $= \mathbf{57.6\,\text{GB}}$, storing all checkpoints on disk risks exhausting cluster storage quotas.

```text
GPU Worker: Finishes Module i
   │
   ├── Writes checkpoint to results/bert_attention_zoo/module_i/
   └── Submits task to AsyncHFHubUploader (Non-blocking queue)
          │
          ├── GPU IMMEDIATELY begins Module i+1 (Zero GPU Idle Time)
          │
          └── Background Thread:
                 1. Creates/Verifies repo: Hooshaai/bert-{slug}-wiki
                 2. Writes Model Card README.md with empirical metrics
                 3. Calls HfApi.upload_folder(...)
                 4. Confirms remote commit hash
                 5. WRITES upload_success.json
                 6. DELETES local model.safetensors / pytorch_model.bin
                 Result: Active local disk space remains strictly < 2 GB!
```

### Dual-Mode Execution Support
1. **In-Process Daemon:** Spawns via `concurrent.futures.ThreadPoolExecutor` directly inside `bert_zoo_train_wiki.py`.
2. **Login-Node Watcher (`bert_zoo_upload_hf.py`):** If PBS compute nodes have restricted outbound internet access, the watcher runs on the login node polling `results/bert_attention_zoo/` for completed modules and uploading them independently.

---

## 7. Self-Contained Model Export (`auto_map` & `modeling_neuralops_bert.py`)

To ensure checkpoints are universally usable without installing NeuralOps, every uploaded model includes:

### `config.json` Patch
```json
{
  "architectures": ["NeuralOpsBertForMaskedLM"],
  "model_type": "bert",
  "auto_map": {
    "AutoConfig": "modeling_neuralops_bert.NeuralOpsBertConfig",
    "AutoModel": "modeling_neuralops_bert.NeuralOpsBertModel",
    "AutoModelForMaskedLM": "modeling_neuralops_bert.NeuralOpsBertForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_neuralops_bert.NeuralOpsBertForSequenceClassification"
  },
  "neuralops_attention_type": "gated_delta_net"
}
```

### Self-Contained `modeling_neuralops_bert.py`
Contains the universal adapter shim, allowing downstream users to load the model with native Hugging Face code:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "Hooshaai/bert-gated-delta-net-wiki",
    trust_remote_code=True,
)
```

---

## 8. Automated Model Card Specification (`README.md`)

Every repository on `Hooshaai` receives an auto-generated Model Card:

```markdown
---
language:
  - en
license: mit
library_name: neuralops
base_model: bert-base-uncased
datasets:
  - wikitext
metrics:
  - perplexity
  - accuracy
tags:
  - neuralops
  - bert
  - attention-zoo
  - knowledge-distillation
  - subquadratic-attention
  - {attn_key}
pipeline_tag: fill-mask
---

# 🧠 BERT-Base ({attn_key}) — Distilled on WikiText-2

This repository hosts a pretrained variant of **`bert-base-uncased`** where all 12 self-attention layers have been replaced with the **`{attn_key}`** mechanism from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps).

Distilled on **WikiText-2** using the **NeuralOps 3-Stage GPU-Saturated Knowledge Distillation Engine** on an NVIDIA RTX 3090.

---

## 📊 Empirical Evaluation & Distillation Benchmark

Evaluated using **Deterministic Masked Language Modeling Pseudo-Perplexity ($\text{MLM-PPL}$)** (15% token masking, `seed=42`):

| Evaluation Phase | MLM Perplexity ($\text{PPL}$) | Masked Token Acc (%) | Delta vs Baseline |
|---|:---:|:---:|:---:|
| **Teacher BERT Baseline (`bert-base-uncased`)** | `{teacher_ppl:.2f}` | `{teacher_acc:.2f}%` | *Reference* |
| **Stage 0 (Closed-Form Ridge Calibration)** | `{pre_kd_ppl:.2f}` | `{pre_kd_acc:.2f}%` | `{pre_delta:+.2f}` |
| **Stage 2 (Post-KD Distilled Model)** | **`{post_kd_ppl:.2f}`** | **`{post_kd_acc:.2f}%`** | **`{post_delta:+.2f}`** |

### Telemetry & Hardware Footprint
- **Hardware:** NVIDIA RTX 3090 24GB (Sharif HPC Cluster)
- **Training Duration:** `300.0s (5.0 minutes)`
- **Peak VRAM:** `{peak_vram_mb} MB`
- **Throughput:** `{tokens_per_sec:.1f} tokens/sec`
- **Distillation Stages:** Stage 0 (Ridge Alignment) $\rightarrow$ Stage 1 (Frozen Subspace MSE/CKA) $\rightarrow$ Stage 2 (Differential-LR Polish)

---

## 🚀 Usage

```python
import neuralops

model, tokenizer = neuralops.load_pretrained_model(
    attention_type="{attn_key}",
    task="masked-lm",
)
```
```

---

## 9. Public APIs & CLI Command Interfaces

### Python API: `neuralops.load_pretrained_model`
```python
def load_pretrained_model(
    attention_type: str,
    model_name: str = "bert-base-uncased",
    task: str = "masked-lm",
    num_labels: int = 2,
    hf_org: str = "Hooshaai",
    device: str = "auto",
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """
    Downloads and instantiates a pretrained attention model from Hugging Face Hub.
    Gracefully falls back to local surgery if offline.
    """
```

### CLI Interface: `neuralops hub`
```bash
# List all 131 models, PPL scores, and download statuses
neuralops hub list

# Pull a specific attention checkpoint locally
neuralops hub pull --attention gated_delta_net --task masked-lm

# Launch the automated 5-minute training and upload engine
neuralops hub train-zoo --minutes-per-module 5.0 --upload-hf --hf-org Hooshaai
```

---

## 10. Sharif HPC Production PBS Execution Job

**File:** `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs`

```bash
#!/bin/bash
#PBS -N bert_zoo_131_wiki
#PBS -q N3090
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=14:00:00
#PBS -l mem=48gb
#PBS -o /home/tahamajlesi.ut/NeuralOps/results/bert_attention_zoo/pbs_stdout.log
#PBS -e /home/tahamajlesi.ut/NeuralOps/results/bert_attention_zoo/pbs_stderr.log

set -euo pipefail

WORK_DIR="/home/tahamajlesi.ut/NeuralOps"
CONDA_ENV="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113"
PYTHON="${CONDA_ENV}/bin/python3"
OUT_DIR="${WORK_DIR}/results/bert_attention_zoo"

mkdir -p "${OUT_DIR}/upload_logs"
cd "${WORK_DIR}"

echo "=== System Telemetry ==="
hostname && date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
${PYTHON} -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"

# Check token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is unset in environment. Checkpoints will be staged locally." >&2
fi

echo "=== Phase 0: Dynamic Key Discovery ==="
${PYTHON} scripts/benchmarks/bert_zoo_discover_keys.py \
    --output "${OUT_DIR}/registry_keys.json" \
    --skipped_output "${OUT_DIR}/skipped_keys.json" \
    --device cuda

echo "=== Phase 1: Baseline PPL Evaluation ==="
${PYTHON} scripts/benchmarks/bert_zoo_baseline_ppl.py \
    --model      bert-base-uncased \
    --dataset    wikitext-2-raw-v1 \
    --seq_len    128 \
    --batch_size 128 \
    --output     "${OUT_DIR}/baseline_ppl.json" \
    --device     cuda

echo "=== Phase 2+3: 5-Minute KD Engine + Async Upload ==="
${PYTHON} scripts/benchmarks/bert_zoo_train_wiki.py \
    --keys_json          "${OUT_DIR}/registry_keys.json" \
    --model              bert-base-uncased \
    --dataset            wikitext-2-raw-v1 \
    --seq_len            128 \
    --batch_size         128 \
    --grad_accum         2 \
    --train_secs         300 \
    --out_dir            "${OUT_DIR}" \
    --metrics_log        "${OUT_DIR}/metrics.jsonl" \
    --hf_org             Hooshaai \
    --hf_upload          1 \
    --keep_local_weights 0 \
    --device             cuda

echo "=== Completed Successfully at $(date) ==="
```

---

## 11. Gradio Interactive Model Arena (`Hooshaai/bert-attention-zoo-demo`)

**Deployment:** Hugging Face Spaces (`spaces/bert_attention_demo/app.py`)
**Features:**
- Dropdown selector for all 131 attention mechanisms.
- Live `[MASK]` token probability inspector.
- Real-time side-by-side comparison: Vanilla BERT PPL vs Pre-KD PPL vs Post-KD PPL.
- Instant copy-paste Python snippet for downstream fine-tuning.

---

## 12. Registry Manifest: All 131 Canonical Attention Units

```text
[001-010] adaptive_span_attention, adaptive_sparse_attention, adaptive_svd_attention, adasplash_attention, alibi,
          asvd_attention, bam_umoe, block_transformer_hiba, chela_slay, cluster_attention
[011-020] compressed_attention, compressed_mha_attention, cosformer, cross_attention, decay_linear_attention,
          delta_product_attention, dynamic_conv_attention, eat, expert_choice_moe, factorised_matrix_state
[021-030] flash_attention3, flash_attention3_fp8, full_attention, fused_svd_kernel, fusion_attn,
          galerkin_attention, gated_delta_net, gated_deltanet_block, gated_hybrid_kernel_attention, gdn2_attention
[031-040] gla_attention, goldfinch_ahn, gqa_attention, granite_hybrid, griffin_ssd,
          halo_regla, hamming_attention, hgrn2_gated_deltanet, hgrn2_gated_layer, hgrn2_gated_outer
[041-050] hla_attention, hybrid_attention, hybrid_local_kernel, hymba_transx, jamba,
          jamba_info, kda_attention, landmark_local_attention, linformer_attention, lola_attention
[051-060] lora_attention, lsh_attention, mamba_block, mamba_svd_flash, minimal_nova,
          minimal_svd, mixattention_moska, mla_attention, moba_attention, moe_deltakv_paged
[061-070] mqa, naive_svd, nasc_attention, nexus, nha_attention,
          nsa_attention, nystrom_attention, optimized_hybrid_attention, orion, padre_attention
[071-080] performer_attention, phantom, positive_kernel_linear_attention, preconditioned_gdn_attention, primal_attention,
          progressive_token_pruning, projection_factorized_attn, qcqa_attention, rala_attention, reap_moe
[081-090] refined_landmark_attention, regla_attention, residual_svd_attention, routing_attention, rpc_attention,
          salt_attention, samba_m3, samba_moe, selective_ssm, simple_ssm
[091-100] sink_attention, soa_attn, spectraformer_attention, spectron, spectron_fp8,
          split_head_hybrid, svd_attention, svd_compressed, svd_linear_attention, svd_linear_decay
[101-110] svd_llm_v2_attention, svd_mamba, svd_projection_attention, token_compressed_attention, tri_branch_hybrid,
          triton_cosformer, triton_svd_linear_attention, ttt_linear_attention, unified_rope, upi_stuffed_mamba
[111-115] vig_attention, vspec, wgqa_attention, xattention, zeros_attention
[116-131] Canonical Extensible Units:
          116: chunk_gla_attention, 117: multi_scale_retention, 118: rwkv6_linear_attention, 119: hgrn_linear_attention,
          120: lightning_attention2, 121: focused_linear_attention, 122: flash_linear_attention, 123: gated_linear_attention_v2,
          124: polynomial_kernel_attention, 125: relu_linear_attention, 126: elu_linear_attention, 127: taylor_series_linear_attention,
          128: randomized_svd_attention, 129: truncated_svd_attention, 130: orthogonal_svd_attention, 131: fused_recurrent_attention
```

---

## 13. Failure Policies, Recovery Matrix & Verification Protocol

| Fault Condition | Automated Handling & Self-Healing |
|---|---|
| **OOM on Attention $i$** | Intercept `torch.cuda.OutOfMemoryError`, empty VRAM cache, halve batch size ($128 \rightarrow 64$), double accumulation ($2 \rightarrow 4$), retry once. |
| **HPC Outbound Network Blocked** | PBS node stages weights locally; login-node watcher daemon detects `training_metrics.json` and performs upload asynchronously. |
| **Walltime Preemption** | Resume capability: engine scans `training_metrics.json` and skips previously completed checkpoints upon re-launch. |
| **Divergent Loss / NaN** | Flagged as `FAIL_NUMERICS` in `metrics.jsonl`; process moves cleanly to module $i+1$. |
| **Disk Quota Constraint** | Local `.safetensors` deleted immediately upon upload commit verification; active disk usage $< 2\text{ GB}$. |

---

## Master Engineering Checklist

- [x] **Master Architectural Plan Documented:** `docs/BERT_ATTENTION_ZOO_TRAINING_PLAN.md`
- [x] **Universal BertAttentionAdapter:** `neuralops/layers/bert_adapter.py`
- [x] **Hub Package & Client APIs:** `neuralops/hub/__init__.py`, `neuralops/hub/bert_zoo.py`, `neuralops/hub/uploader.py`
- [x] **Dynamic Key Discovery Script:** `scripts/benchmarks/bert_zoo_discover_keys.py`
- [x] **Deterministic 15% MLM PPL Evaluator:** `scripts/benchmarks/bert_zoo_baseline_ppl.py`
- [x] **5-Minute 100% GPU-Saturated KD Engine:** `scripts/benchmarks/bert_zoo_train_wiki.py`
- [x] **Dual-Mode Watcher Uploader:** `scripts/hpc/bert_zoo_upload_hf.py`
- [x] **Production PBS Job Script:** `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs`
- [x] **Interactive Gradio Space:** `spaces/bert_attention_demo/app.py`
- [x] **Unit & Integration Test Suite:** `tests/test_bert_hf_hub_zoo.py`
- [x] **README.md Ecosystem Documentation:** Updated with Model Zoo badges & 1-line usage recipes

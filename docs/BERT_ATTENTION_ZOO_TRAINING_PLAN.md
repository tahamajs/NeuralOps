# BERT Attention Zoo — WikiText Training Plan
## Full-Pipeline: 115 NeuralOps Attentions × 5-Min KD × HuggingFace Upload × Gradio Demo

**Version:** 2.0.0 (Incorporating Mandatory Refinements)  
**Date:** 2026-11-09  
**Author:** Taha Majlesi  
**Target Architecture:** `bert-base-uncased` (110M params, 768-dim, 12-heads, 12 layers)  
**Distillation Corpus:** `WikiText-2` (Masked Language Modeling, Sequence Length = 128/256)  
**Primary Compute Host:** Sharif HPC Cluster (`GPU-3090-1`, NVIDIA RTX 3090 24GB GDDR6X, CUDA 11.3)  
**Hugging Face Hub Organization:** `Hooshaai` (Primary) / `tahamajs` (Mirror)  
**Estimated Wall-clock:** ~11.5 hours (115 models × 5 min + background upload overhead)

---

## 5 Mandatory Engineering Refinements (Canonical Directives)

1. **Proper BERT Surgery & Weight Inheritance (`replace_attention`)**:
   - In Stage 0, do NOT assign `layer.attention.self = create_attention(...)` raw without wrapping/adapting signature and output projection. Use our tested `replace_attention(student, attention=attn_key, copy_weights=True)` from `neuralops.api` (or dedicated `BertNeuralOpsAttentionWrapper` that inherits pretrained $W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O$ weights from the teacher and matches `BertAttention`'s tuple return signature).
   - In all scripts and documentation, rely dynamically on `registry_keys.json` discovered directly from `ATTENTION_REGISTRY` (never hard-code duplicate numbering).

2. **Deterministic 15% Masked LM Perplexity (`eval_ppl`)**:
   - Because BERT is a bidirectional encoder, evaluating `BertForMaskedLM` with unmasked `input_ids == labels` leaks the target token into self-attention and yields trivial $\text{PPL} \approx 1.0$.
   - In both `bert_zoo_baseline_ppl.py` and `bert_zoo_train_wiki.py` (`eval_ppl`), apply a deterministic 15% token mask (seeded with `torch.Generator().manual_seed(42)`) where non-masked positions have `labels = -100`, computing true Masked LM Pseudo-Perplexity $\text{PPL}_{\text{MLM}} = \exp(\mathcal{L}_{\text{MLM}})$ and Masked Token Accuracy ($\text{Acc}_{\text{MLM}}$).

3. **True 24 GB RTX 3090 VRAM Saturation & Fast Tensor-Core Pillar 4**:
   - Instead of leaving 18.5 GB of the 24 GB RTX 3090 idle at `batch_size=64`, pre-tokenize WikiText-2 into RAM/VRAM contiguous tensors and start at `batch_size=128` (or `192`, with `grad_accum=2`) with automatic OOM halving, ensuring 18–20 GB of VRAM and 100% of Tensor Cores are utilized during the 300s window.
   - In `compute_6pillar_loss`, compute Pillar 4 (SVD Spectrum alignment) either every 10 steps OR via fast normalized Gram-matrix covariance matching ($G = \frac{1}{S} X X^\top$) in FP16 Tensor Cores so `torch.linalg.svdvals` never stalls the GPU pipeline on every micro-batch.

4. **HPC Disk Quota Protection & Dual-Mode HF Upload**:
   - 115 BERT checkpoints $\times$ 440 MB = ~51 GB, which can exceed `/home` disk quota on HPC.
   - In `upload_to_hf`, as soon as `api.upload_folder` succeeds, write `upload_success.json` and **delete the heavy `pytorch_model.bin` / `model.safetensors` file from local disk** (unless `--keep_local_weights 1` is passed).
   - For resume logic in `bert_zoo_train_wiki.py`, check `(out_dir / attn_key / "training_metrics.json").exists()` instead of `pytorch_model.bin`.
   - Dual-Mode Upload: Allow `scripts/hpc/bert_zoo_upload_hf.py` to run in `--watch_dir results/bert_attention_zoo` mode on the HPC **login node** in case the PBS compute node has restricted outbound internet!

5. **Self-Contained Hugging Face Hub Loading (`auto_map` + `neuralops.load_bert`)**:
   - Save a lightweight `modeling_neuralops_bert.py` + `auto_map` entry in `config.json` (and `neuralops_config.json`) inside each checkpoint directory so `AutoModelForMaskedLM.from_pretrained("Hooshaai/bert-{slug}-wiki", trust_remote_code=True)` in the Model Card and Gradio Space loads the exact custom attention architecture and trained weights without missing-key warnings!

---

## Table of Contents

1. [Overview & Summary](#1-overview--summary)
2. [Dynamic Registry Discovery](#2-dynamic-registry-discovery)
3. [RTX 3090 Hardware & 20GB VRAM Saturation Budget](#3-rtx-3090-hardware--20gb-vram-saturation-budget)
4. [3-Stage KD Curriculum & Fast Tensor-Core Pillars](#4-3-stage-kd-curriculum--fast-tensor-core-pillars)
5. [Phase 0 — Environment & Dynamic Key Discovery](#phase-0--environment--dynamic-key-discovery)
6. [Phase 1 — Deterministic 15% Masked LM Baseline PPL](#phase-1--deterministic-15-masked-lm-baseline-ppl)
7. [Phase 2 — Per-Attention Training Engine (`replace_attention`)](#phase-2--per-attention-training-engine-replace_attention)
8. [Phase 3 — Dual-Mode HF Hub Upload & Disk Protection](#phase-3--dual-mode-hf-hub-upload--disk-protection)
9. [Phase 4 — Self-Contained Model Cards (`auto_map`)](#phase-4--self-contained-model-cards-auto_map)
10. [Phase 5 — Gradio Space Interface](#phase-5--gradio-space-interface)
11. [Phase 6 — Production PBS Script for Sharif HPC](#phase-6--production-pbs-script-for-sharif-hpc)
12. [Phase 7 — README & Documentation](#phase-7--readme--documentation)
13. [File Map & Manifest](#13-file-map--manifest)
14. [Failure Policy & Recovery](#14-failure-policy--recovery)
15. [Verification Checklist](#15-verification-checklist)

---

## 1. Overview & Summary

This pipeline trains **BERT-base-uncased** on **WikiText-2** using a high-throughput, 6-Pillar Knowledge Distillation pipeline for each of the **115 confirmed NeuralOps attention mechanisms**.

```text
Confirmed Registry Keys: 115 attention mechanisms
Training Window:         Strictly 300 seconds (5 minutes) per attention
Evaluation Metric:       Deterministic 15% Masked LM Pseudo-PPL & Accuracy
Storage Management:      Auto-Cleanup Queue (Active disk < 2 GB; deletes local .safetensors after upload)
Upload Target:           Hugging Face Hub: Hooshaai/bert-{attn_slug}-wiki
Hub Autoloading:         auto_map in config.json + lightweight modeling_neuralops_bert.py
Walltime Budget:         115 × 5 min ≈ 9.6 h + 1.5 h overhead = ~11.1 h (PBS walltime: 14:00:00)
```

### Stage Timeline per Module (Total = 300s)

| Stage | Name | Window | Parameters Trainable | Optimizer & LR | Primary Losses |
|---|---|---|---|---|---|
| **0** | Closed-Form Ridge Calibration | $t=0\dots 5\,\text{s}$ | None (analytic solve) | — | $W_{\text{align}} = (Z^\top Z + \lambda I)^{-1} Z^\top Y$ |
| **Pre**| Deterministic MLM Evaluation | $\sim 5\,\text{s}$ | None (eval mode) | — | Record `pre_kd_ppl` and `pre_kd_acc` |
| **1** | Operator Subspace Mimicry | $t=5\dots 95\,\text{s}$ (30%) | Attention & LayerNorms only | AdamW $\text{lr}=1.0\times 10^{-3}$ | Hidden MSE + Linear CKA + MLM KL |
| **2** | End-to-End KD Polish | $t=95\dots 300\,\text{s}$ (70%)| Full Model Unfrozen | Differential $\text{lr}=5\times 10^{-4} / 5\times 10^{-5}$ | Fused KL + MLM Cross-Entropy |
| **Post**| Final Deterministic Evaluation| $\sim 5\,\text{s}$ | None (eval mode) | — | Record `post_kd_ppl`, `post_kd_acc`, VRAM, Speed |

---

## 2. Dynamic Registry Discovery

Rather than relying on static or miscounted numbers, the system executes dynamic discovery via `scripts/benchmarks/bert_zoo_discover_keys.py`.

The 115 registered keys in `ATTENTION_REGISTRY` include:

```text
adaptive_span_attention, adaptive_sparse_attention, adaptive_svd_attention, adasplash_attention, alibi,
asvd_attention, bam_umoe, block_transformer_hiba, chela_slay, cluster_attention, compressed_attention,
compressed_mha_attention, cosformer, cross_attention, decay_linear_attention, delta_product_attention,
dynamic_conv_attention, eat, expert_choice_moe, factorised_matrix_state, flash_attention3,
flash_attention3_fp8, full_attention, fused_svd_kernel, fusion_attn, galerkin_attention, gated_delta_net,
gated_deltanet_block, gated_hybrid_kernel_attention, gdn2_attention, gla_attention, goldfinch_ahn,
gqa_attention, granite_hybrid, griffin_ssd, halo_regla, hamming_attention, hgrn2_gated_deltanet,
hgrn2_gated_layer, hgrn2_gated_outer, hla_attention, hybrid_attention, hybrid_local_kernel,
hymba_transx, jamba, jamba_info, kda_attention, landmark_local_attention, linformer_attention,
lola_attention, lora_attention, lsh_attention, mamba_block, mamba_svd_flash, minimal_nova,
minimal_svd, mixattention_moska, mla_attention, moba_attention, moe_deltakv_paged, mqa, naive_svd,
nasc_attention, nexus, nha_attention, nsa_attention, nystrom_attention, optimized_hybrid_attention,
orion, padre_attention, performer_attention, phantom, positive_kernel_linear_attention,
preconditioned_gdn_attention, primal_attention, progressive_token_pruning, projection_factorized_attn,
qcqa_attention, rala_attention, reap_moe, refined_landmark_attention, regla_attention,
residual_svd_attention, routing_attention, rpc_attention, salt_attention, samba_m3, samba_moe,
selective_ssm, simple_ssm, sink_attention, soa_attn, spectraformer_attention, spectron,
spectron_fp8, split_head_hybrid, svd_attention, svd_compressed, svd_linear_attention,
svd_linear_decay, svd_llm_v2_attention, svd_mamba, svd_projection_attention,
token_compressed_attention, tri_branch_hybrid, triton_cosformer, triton_svd_linear_attention,
ttt_linear_attention, unified_rope, upi_stuffed_mamba, vig_attention, vspec, wgqa_attention,
xattention, zeros_attention.
```

The runner loads `results/bert_attention_zoo/registry_keys.json` generated dynamically in Phase 0.

---

## 3. RTX 3090 Hardware & 20GB VRAM Saturation Budget

### VRAM Allocation Model (24,576 MB Total GDDR6X)

To avoid idling 18.5 GB of VRAM, the training script uses `InMemoryWikiTextDataset` and a saturation batch size:

| Subsystem | Memory Footprint | Description |
|---|---|---|
| **Teacher BERT Cache** | ~2,500 MB | Cached hidden states & logits for training chunks in FP16 |
| **Student BERT Weights** | ~440 MB (FP32) | Unfrozen student model |
| **Optimizer States (AdamW)** | ~880 MB | Momentum + Variance buffers |
| **Gradients** | ~440 MB | FP32 gradient buffers |
| **In-VRAM Token Tensor** | ~60 MB | Full WikiText-2 tokenized chunk array |
| **Activation Cache ($B=128, S=128$)** | ~14,000 MB | 12 transformer layers, intermediate states |
| **Total Peak Working Memory** | **~18,320 MB** | **~75% to 80% VRAM Saturation** |
| **Headroom Buffer** | **~6,256 MB** | Prevents OOM on attention modules with larger state sizes |

### Dynamic Batch Saturation Policy

```python
# Initial high-throughput configuration:
BATCH_SIZE = 128      # Starts at 128 (or 192 if supported)
GRAD_ACCUM = 2        # Effective batch size = 256
SEQ_LEN    = 128      # Standard BERT fine-tuning length (or 256)
AMP_DTYPE  = torch.float16
```

If an attention module raises `torch.cuda.OutOfMemoryError`, the exception handler:
1. Calls `torch.cuda.empty_cache()` and `gc.collect()`.
2. Halves the batch size: $B \leftarrow B // 2$ (e.g., $128 \rightarrow 64$).
3. Doubles gradient accumulation: $\text{accum} \leftarrow \text{accum} \times 2$ (maintaining effective batch size = 256).
4. Retries the module once.

---

## 4. 3-Stage KD Curriculum & Fast Tensor-Core Pillars

### 4.1. Fast Normalized Gram-Matrix Alignment (Pillar 4 Optimization)

Standard singular value decomposition (`torch.linalg.svdvals`) forces a CPU-GPU synchronization barrier and runs slow iterative algorithms, stalling Tensor Cores on every micro-batch.

**Refinement:** We replace full SVD decomposition on every micro-batch with **Fast Normalized Gram Covariance Matching** in FP16:

$$\mathcal{L}_{\text{Cov}} = \frac{1}{D^2} \left\| \frac{h_S^\top h_S}{\|h_S\|_F} - \frac{h_T^\top h_T}{\|h_T\|_F} \right\|_F^2$$

Full `torch.linalg.svdvals` is evaluated strictly once every 20 steps (or in eval phases) to monitor spectral decay, while the fast Gram loss runs on every step with 100% Tensor Core speed.

### 4.2. 6-Pillar Loss Composition

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{Logits}} + w_2 \mathcal{L}_{\text{Hidden-MSE}} + w_3 \mathcal{L}_{\text{Hidden-Cos}} + w_4 \mathcal{L}_{\text{Cov}} + w_5 \mathcal{L}_{\text{Fisher}} + w_6 \mathcal{L}_{\text{Sparsity}} + w_{\text{MLM}} \mathcal{L}_{\text{MLM-CE}}$$

- $w_1 = 0.50$ (Logits KL with temperature $T=2.0$)
- $w_2 = 0.30$ (Layerwise Hidden MSE)
- $w_3 = 0.20$ (Layerwise Hidden Cosine Alignment)
- $w_4 = 0.15$ (Fast Gram Subspace Covariance)
- $w_5 = 0.10$ (Fisher Curvature Alignment)
- $w_6 = 0.05$ (Attention Sparsity / Entropy)
- $w_{\text{MLM}} = 0.30$ (Ground-truth Masked Language Modeling CE in Stage 2)

---

## Phase 0 — Environment & Dynamic Key Discovery

**Script:** `scripts/benchmarks/bert_zoo_discover_keys.py`

This script probes every registered attention mechanism by instantiating it with BERT's dimensions (`embed_dim=768, num_heads=12`), wrapping it with `BertAttentionAdapter`, and verifying a forward pass.

```bash
python3 scripts/benchmarks/bert_zoo_discover_keys.py \
    --output results/bert_attention_zoo/registry_keys.json \
    --skipped_output results/bert_attention_zoo/skipped_keys.json \
    --device cuda
```

Output: Validated `registry_keys.json` with all instantiable keys.

---

## Phase 1 — Deterministic 15% Masked LM Baseline PPL

**Script:** `scripts/benchmarks/bert_zoo_baseline_ppl.py`

Evaluates unmodified `bert-base-uncased` on `wikitext-2-raw-v1` (validation split) with a deterministic 15% token mask:

```bash
python3 scripts/benchmarks/bert_zoo_baseline_ppl.py \
    --model bert-base-uncased \
    --dataset wikitext-2-raw-v1 \
    --seq_len 128 \
    --batch_size 128 \
    --output results/bert_attention_zoo/baseline_ppl.json \
    --device cuda
```

Output file `baseline_ppl.json`:
```json
{
  "ppl_vanilla_bert": 6.84,
  "acc_vanilla_bert": 68.42,
  "loss": 1.9228,
  "num_tokens_masked": 37620,
  "model": "bert-base-uncased",
  "dataset": "wikitext-2-raw-v1"
}
```

---

## Phase 2 — Per-Attention Training Engine (`replace_attention`)

**Script:** `scripts/benchmarks/bert_zoo_train_wiki.py`

### Implementation Highlights

1. **Surgery via `replace_attention(model, attention=attn_key, copy_weights=True)`**:
   Inherits query, key, value, and dense output weights.
2. **In-VRAM Dataset & Pre-Cached Teacher**:
   Loads the dataset directly into GPU memory tensors; evaluates teacher representations once.
3. **Resume Logic via `training_metrics.json`**:
   Checks `if (out_dir / attn_key / "training_metrics.json").exists(): continue`.
4. **Execution Command**:

```bash
python3 scripts/benchmarks/bert_zoo_train_wiki.py \
    --keys_json          results/bert_attention_zoo/registry_keys.json \
    --model              bert-base-uncased \
    --dataset            wikitext-2-raw-v1 \
    --seq_len            128 \
    --batch_size         128 \
    --grad_accum         2 \
    --train_secs         300 \
    --out_dir            results/bert_attention_zoo \
    --metrics_log        results/bert_attention_zoo/metrics.jsonl \
    --hf_org             Hooshaai \
    --hf_upload          1 \
    --keep_local_weights 0 \
    --device             cuda
```

---

## Phase 3 — Dual-Mode HF Hub Upload & Disk Protection

**Script:** `scripts/hpc/bert_zoo_upload_hf.py`

### Mode 1: Inline Asynchronous Daemon (inside training script)
As each module completes, a background worker uploads the directory and deletes the local `.safetensors` file upon verification.

### Mode 2: Standalone Watcher (for login node with internet)
If compute nodes lack outbound internet access:

```bash
# Executed on login node:
python3 scripts/hpc/bert_zoo_upload_hf.py \
    --watch_dir results/bert_attention_zoo \
    --hf_org Hooshaai \
    --cleanup 1 \
    --poll_interval 30
```

The watcher scans `results/bert_attention_zoo/` for completed `training_metrics.json` without an `upload_success.json`, uploads the folder to Hugging Face Hub, creates `upload_success.json`, and deletes `model.safetensors` to protect disk quota.

---

## Phase 4 — Self-Contained Model Cards (`auto_map`)

Inside each model staging directory, the following files are written:

1. `modeling_neuralops_bert.py`: Self-contained class extending `BertForMaskedLM` that integrates `BertAttentionAdapter` and the specific attention mechanism.
2. `config.json` containing `auto_map`:
```json
{
  "architectures": ["NeuralOpsBertForMaskedLM"],
  "model_type": "bert",
  "auto_map": {
    "AutoConfig": "modeling_neuralops_bert.NeuralOpsBertConfig",
    "AutoModelForMaskedLM": "modeling_neuralops_bert.NeuralOpsBertForMaskedLM"
  },
  "neuralops_attention_type": "gated_delta_net"
}
```
3. `README.md`: Auto-generated Model Card with quantitative PPL and accuracy comparison table.

This guarantees that:
```python
model = AutoModelForMaskedLM.from_pretrained("Hooshaai/bert-gated-delta-net-wiki", trust_remote_code=True)
```
works seamlessly anywhere without installing NeuralOps from source.

---

## Phase 5 — Gradio Space Interface

**Path:** `spaces/bert_attention_demo/app.py`  
**Target:** `Hooshaai/bert-attention-zoo-demo`

Interactive UI features:
- Attention mechanism dropdown (115 models).
- Interactive fill-mask prompt: `"NeuralOps makes deep learning models [MASK] and efficient."`
- Real-time prediction display with top-5 token probabilities.
- Model telemetry: Pre-KD PPL vs Post-KD PPL vs Vanilla BERT Baseline.
- Direct links to Model Cards on Hugging Face Hub.

---

## Phase 6 — Production PBS Script for Sharif HPC

**Path:** `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs`

```bash
#!/bin/bash
#PBS -N bert_zoo_115_wiki
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

echo "=== Environment Info ==="
hostname && date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
${PYTHON} -c "import torch; print('PyTorch', torch.__version__); print('CUDA Device:', torch.cuda.get_device_name(0))"

# Check HF Token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is unset. Uploads will be staged locally." >&2
fi

echo "=== Phase 0: Dynamic Key Discovery ==="
${PYTHON} scripts/benchmarks/bert_zoo_discover_keys.py \
    --output "${OUT_DIR}/registry_keys.json" \
    --skipped_output "${OUT_DIR}/skipped_keys.json" \
    --device cuda

echo "=== Phase 1: Baseline PPL ==="
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

echo "=== Complete at $(date) ==="
```

---

## Phase 7 — README & Documentation

After completing training and verifying uploads, update the main `README.md` with:
- Model Zoo overview badge linking to `Hooshaai`.
- Leaderboard table ranking top attention mechanisms by MLM Perplexity reduction on WikiText-2.
- 1-line usage instructions (`neuralops.load_pretrained_bert`).

---

## 13. File Map & Manifest

```text
NeuralOps/
├── neuralops/
│   ├── layers/
│   │   └── bert_adapter.py               ← Universal BertAttentionAdapter wrapper
│   └── hub/
│       ├── __init__.py                   ← High-level hub package exports
│       ├── bert_zoo.py                   ← load_pretrained_bert & list_pretrained_bert_models
│       └── uploader.py                   ← AsyncHFHubUploader with auto-cleanup
├── scripts/
│   ├── benchmarks/
│   │   ├── bert_zoo_discover_keys.py     ← Phase 0: Dynamic key discovery
│   │   ├── bert_zoo_baseline_ppl.py      ← Phase 1: Deterministic 15% MLM baseline PPL
│   │   └── bert_zoo_train_wiki.py        ← Phase 2: Master 5-minute KD engine
│   └── hpc/
│       ├── bert_zoo_upload_hf.py         ← Phase 3: Standalone login-node watcher
│       └── job_bert_131_wikitext_hf_zoo.pbs ← Phase 6: Production PBS job script
├── spaces/
│   └── bert_attention_demo/
│       ├── app.py                        ← Phase 5: Gradio interactive web UI
│       └── requirements.txt
├── docs/
│   └── BERT_ATTENTION_ZOO_TRAINING_PLAN.md ← This master plan document
└── results/
    └── bert_attention_zoo/
        ├── registry_keys.json            ← Dynamically discovered keys
        ├── baseline_ppl.json             ← Vanilla BERT baseline metrics
        ├── metrics.jsonl                 ← Master append-only evaluation ledger
        └── {attn_key}/                   ← Staging directory (weights deleted after upload)
            ├── config.json               ← Contains auto_map
            ├── modeling_neuralops_bert.py← Self-contained architecture definition
            ├── README.md                 ← Auto-generated Model Card
            ├── training_metrics.json     ← Empirical evaluation results
            └── upload_success.json       ← Upload confirmation & commit hash
```

---

## 14. Failure Policy & Recovery

| Failure Condition | Automatic Recovery Procedure |
|---|---|
| **OOM on Module $i$** | Catch exception, empty GPU cache, halve batch size ($128 \rightarrow 64$), double gradient accumulation ($2 \rightarrow 4$), retry. |
| **Outbound Network Blocked on PBS Node** | Training writes local checkpoints; standalone `bert_zoo_upload_hf.py` runs concurrently on login node watching output directory. |
| **PBS Job Time Exceeded** | Resume feature: script checks `training_metrics.json` and skips finished modules on restart. |
| **NaN / Divergent Loss** | Flagged as `FAIL_NUMERICS` in metrics ledger; pipeline proceeds to module $i+1$ without crashing. |
| **Disk Space Constraint** | `model.safetensors` deleted immediately upon upload verification; disk usage remains $< 2\text{ GB}$. |

---

## 15. Verification Checklist

- [x] **Master Plan Updated:** All 5 mandatory engineering refinements documented in `docs/BERT_ATTENTION_ZOO_TRAINING_PLAN.md`
- [ ] **Universal Adapter:** `neuralops/layers/bert_adapter.py` created and tested
- [ ] **Discovery Script:** `scripts/benchmarks/bert_zoo_discover_keys.py` verified
- [ ] **Baseline Script:** `scripts/benchmarks/bert_zoo_baseline_ppl.py` verified
- [ ] **Master KD Script:** `scripts/benchmarks/bert_zoo_train_wiki.py` created with fast Gram covariance & batch saturation
- [ ] **Upload Daemon & Watcher:** `scripts/hpc/bert_zoo_upload_hf.py` created with disk cleanup
- [ ] **End-User Hub API:** `neuralops/hub/bert_zoo.py` created
- [ ] **Gradio Demo Space:** `spaces/bert_attention_demo/app.py` created
- [ ] **PBS Job Configured:** `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs` ready for submission

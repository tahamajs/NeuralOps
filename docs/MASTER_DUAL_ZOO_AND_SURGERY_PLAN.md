# 🏛️ NeuralOps Master Dual-Zoo & Empirical Selection Plan
## Exhaustive Multi-Phase GPU Screening, Surgery Architecture Upgrades, Pareto Selection & End-to-End Distillation for BERT-131 & Qwen-Generative Zoos

**Document ID:** `NOP-MASTER-ZOO-SELECTION-v5.0`
**Status:** Canonical Multi-Phase Empirical Roadmap & Code-Reuse Audit
**Target Backbones:**
  - **Track A (NLU & Encoder Zoo):** `bert-base-uncased` (110M params, 12 layers, 768-dim, 12 heads)
  - **Track B (Generative & Causal LM Zoo):** `Qwen/Qwen2.5-0.5B` (490M params, 24 layers, GQA 14:2 heads, 32k context) & `SmolLM2-135M`
**HPC Compute Target:** Sharif HPC Cluster (`GPU-3090-1`, NVIDIA RTX 3090 24GB GDDR6X, CUDA 11.3, GCC 8.2)
**Distributed Secondary Hosts:** `ssh gpu` (RTX 4070 8GB SM 8.9) & `ssh tahoora` (GTX 1060 6GB SM 6.1)
**Target Hugging Face Hub Organization:** `Hooshaai` (Official Research Laboratory Org)
**Security Policy:** Absolute redaction of `$HF_TOKEN` — injected via environment variables only

---

## Table of Contents

1. [Architectural Overview & Repository Code-Reuse Audit](#1-architectural-overview--repository-code-reuse-audit)
2. [The 5 Critical Surgery Engine Upgrades](#2-the-5-critical-surgery-engine-upgrades)
3. [The Dual-Track Architecture: BERT Encoder Zoo & Qwen Causal LM Roadmap](#3-the-dual-track-architecture-bert-encoder-zoo--qwen-causal-lm-roadmap)
4. [Mathematical Formulation: MLM Pseudo-Perplexity & Metric Integrity](#4-mathematical-formulation-mlm-pseudo-perplexity--metric-integrity)
5. [The 14-Phase Empirical Screening & Distillation Funnel](#5-the-14-phase-empirical-screening--distillation-funnel)
   - [Phase 0: Hardware, Invariants & Registry Inventory Audit](#phase-0-hardware-invariants--registry-inventory-audit)
   - [Phase 1: Core Surgery Engine Upgrade & Single-W_O Contract Enforcement](#phase-1-core-surgery-engine-upgrade--single-w_o-contract-enforcement)
   - [Phase 2: Teacher Baseline Telemetry (BERT 15% MLM PPL & Qwen CLM PPL)](#phase-2-teacher-baseline-telemetry-bert-15-mlm-ppl--qwen-clm-ppl)
   - [Phase 3: GPU Zero-Shot Instantiation & Stability Screening (All 131 Modules)](#phase-3-gpu-zero-shot-instantiation--stability-screening-all-131-modules)
   - [Phase 4: Stage 0 Closed-Form Ridge Calibration Profiling & Metric Delta Recording](#phase-4-stage-0-closed-form-ridge-calibration-profiling--metric-delta-recording)
   - [Phase 5: 60-Second Pilot KD Screening & Multi-Objective Pareto Frontier Analysis](#phase-5-60-second-pilot-kd-screening--multi-objective-pareto-frontier-analysis)
   - [Phase 6: Attention Mechanism Selection & Classification (Tier-1 Champions vs Tier-2 Standard)](#phase-6-attention-mechanism-selection--classification-tier-1-champions-vs-tier-2-standard)
   - [Phase 7: Full 5-Minute (300s) 3-Stage 6-Pillar KD Distillation on Selected Modules](#phase-7-full-5-minute-300s-3-stage-6-pillar-kd-distillation-on-selected-modules)
   - [Phase 8: Asynchronous Background HF Hub Upload & Storage Quota Protection (Hooshaai)](#phase-8-asynchronous-background-hf-hub-upload--storage-quota-protection-hooshaai)
   - [Phase 9: Self-Contained Zero-Dependency Checkpoint Validation (auto_map + modeling shims)](#phase-9-self-contained-zero-dependency-checkpoint-validation-auto_map--modeling-shims)
   - [Phase 10: Track B — Qwen2.5-0.5B Generative Causal LM Distillation Sweep (Constant O(1) KV)](#phase-10-track-b--qwen25-05b-generative-causal-lm-distillation-sweep-constant-o1-kv)
   - [Phase 11: Multi-Format Packaging for Universal Accessibility (GGUF, Ollama, vLLM / SGLang)](#phase-11-multi-format-packaging-for-universal-accessibility-gguf-ollama-vllm--sglang)
   - [Phase 12: Interactive Gradio Attention Arena (Hooshaai/bert-attention-zoo-demo)](#phase-12-interactive-gradio-attention-arena-hooshaaibert-attention-zoo-demo)
   - [Phase 13: End-User API & CLI Integration (neuralops.load_pretrained_model, neuralops hub)](#phase-13-end-user-api--cli-integration-neuralopsload_pretrained_model-neuralops-hub)
   - [Phase 14: Master Benchmark Ledger Synchronization, Paper Tables & CI Release Gates](#phase-14-master-benchmark-ledger-synchronization-paper-tables--ci-release-gates)
6. [Catalog of All 131 Canonical Attention Units](#6-catalog-of-all-131-canonical-attention-units)
7. [Hardware & VRAM Saturation Budget on RTX 3090](#7-hardware--vram-saturation-budget-on-rtx-3090)
8. [Failure Policies, Recovery Matrix & Verification Protocols](#8-failure-policies-recovery-matrix--verification-protocols)

---

## 1. Architectural Overview & Repository Code-Reuse Audit

A forensic analysis of the `NeuralOps` repository reveals **over 1,500 production-grade Python modules** implementing specialized surgery, distillation, memory management, and export capabilities. Rather than reimplementing functionality from scratch, this master plan directly leverages and unifies existing subsystems:

### Existing Subsystem Inventory & Reusable Components

| Subsystem | Existing File / Class | Reusable Functionality for Model Zoo |
|---|---|---|
| **6-Pillar Distillation** | `neuralops/trainer/distillation/multi_level_kd.py`<br>`MultiLevelKDTrainer` | Multi-objective loss suite: Fused Triton KL (`fused_linear_kl_loss`), SmoothMax CKA (`SmoothMaxOverLayersCKASteeringLoss`), MiniLM relation loss (`MiniLMValueRelationKDLoss`), and Attention stability (`AttentionStabilityReg`). |
| **VRAM Saturation & Memory** | `neuralops/trainer/smart_gpu_trainer.py`<br>`AutoMaxGPUTrainer`, `GPUMemoryManager` | Adaptive binary search for maximum batch size (85%–92% VRAM saturation), asynchronous double-buffered CUDA stream prefetching (`AsyncDataPrefetcher`), real-time OOM auto-recovery. |
| **Stage-0 Closed-Form Ridge** | `neuralops/core/_attention_surgery.py`<br>`calibrate_surgically_replaced_model` | Analytical solution to Tikhonov-regularized Ridge problem: $W_{\text{align}} = (Z_c^\top Z_c + \lambda I)^{-1} Z_c^\top Y_c$, folding affine alignment directly into `out_proj.weight` and `out_proj.bias`. |
| **Model Patching & Cache** | `neuralops/evaluation/eval_patcher.py`<br>`apply_module_to_model`, `get_base_model_cached` | Isolated base model templates avoiding Python deepcopy overhead; unified dispatch for both `ATTENTION_REGISTRY` and `COMPRESSOR_REGISTRY` across BERT, RoBERTa, GPT-2, and Qwen. |
| **Universal Forward Adapter** | `neuralops/utils/replace_attention.py`<br>`UniversalAttentionWrapper` | Polymorphic forward adapter normalizing tuple unpacking, past key values, RoPE bypass, and attention mask formatting across Hugging Face model families. |
| **BERT Surgery Adapter** | `neuralops/layers/bert_adapter.py`<br>`BertAttentionAdapter` | Normalizes core attention forward signatures to strict `BertAttention` tuple contract `[B, S, 768] -> (context, attn_weights)`. |
| **Model Family Introspection** | `neuralops/core/model_family.py`<br>`detect_family`, `ModelAdapter` | Automatically detects architecture (BERT, RoBERTa, Qwen, LLaMA, Mistral, GPT-2) and exposes standardized projection names and GQA configurations. |
| **Transactional Surgery** | `neuralops/pipeline_transform/snapshot.py`<br>`ModelSnapshot`, `RollbackError` | In-memory selective model snapshotting; guarantees atomic rollback if surgery or forward/backward gates fail. |
| **PEFT & Fast Healing** | `neuralops/trainer/regimes/dora_healing.py`<br>`neuralops/trainer/regimes/fisher_micro.py` | Direction-magnitude weight decomposition (DoRA) and Fisher information curvature damping for rapid recovery post-surgery. |
| **GGUF / Ollama Export** | `neuralops/export/gguf_exporter.py`<br>`neuralops/export/exporter.py` | Converts trained checkpoints to GGUF format for instant local execution via Ollama without dedicated GPUs. |
| **High-Throughput Serving** | `neuralops/integrations/vllm.py`<br>`neuralops/integrations/sglang.py` | Connects linear attention recurrent state matrices with vLLM PagedAttention and SGLang RadixAttention for production inference. |
| **Asynchronous HF Upload** | `neuralops/hub/uploader.py`<br>`AsyncHFHubUploader` | Non-blocking thread-pool uploader with automated remote commit verification and atomic local `.safetensors` deletion. |

---

## 2. The 5 Critical Surgery Engine Upgrades

Before initiating large-scale training sweeps, five core architectural surgery flaws must be resolved in `neuralops/core/`:

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

### 1. Fix the "Double Output Projection ($W_O$)" Bug in BERT / RoBERTa Surgery
- **The Bug:** In Hugging Face `BertLayer`, `layer.attention` consists of `layer.attention.self` (`BertSelfAttention`, projecting Q, K, V with NO output projection) and `layer.attention.output` (`BertSelfOutput`, which applies `dense` $W_O$, dropout, and LayerNorm). However, NeuralOps attention modules in `ATTENTION_REGISTRY` already bundle their own internal `out_proj`. Injecting a NeuralOps module into `layer.attention.self` caused hidden states to pass through `new_attn.out_proj` AND then through `layer.attention.output.dense`, applying two output projections sequentially and destroying zero-shot representations.
- **The Upgrade:**
  1. Extract $(W_Q, b_Q, W_K, b_K, W_V, b_V)$ from `layer.attention.self` and $(W_O, b_O)$ from `layer.attention.output.dense`.
  2. Transplant all four projections into the new module (`q_proj, k_proj, v_proj, out_proj`).
  3. Replace `layer.attention.output.dense` with `nn.Identity()` so there is **strictly ONE output projection per layer**.
  4. Ensure `UniversalAttentionWrapper` forces `is_causal=False` (bidirectional encoder mode) when the model is an encoder (`not config.is_decoder`), returning `(context_layer,)`.

### 2. Family-Specific Smart Weight Transplantation
During `replace_attention(..., copy_weights=True)`, weight initialization must match the operator's mathematical paradigm:
- **Standard / Linear / Sparse / Gated Modules:** Direct tensor copy. For gated mechanisms (`gated_delta_net`, `gla_attention`, `hgrn2`, `kda_attention`), initialize the forget-gate bias vector $b_g$ such that the initial state retention gate is high:
  $$\sigma(b_g) \approx 0.88 - 0.95 \implies b_g \sim \mathcal{U}(2.0, 3.0)$$
- **SVD & Low-Rank Modules (`svd_linear_attention`, `asvd_attention`, `compressed_mha_attention`):** Perform exact truncated SVD ($\text{SVD}(W) = U \Sigma V^\top$) on the pretrained $W_Q, W_K, W_V, W_O$ matrices at surgery time. Factorize into rank-$r$ factors:
  $$U_r = U_{:, :r} \sqrt{\Sigma_{:r}}, \quad V_r = \sqrt{\Sigma_{:r}} V_{:r, :}^\top$$
- **SSM / Mamba Modules (`mamba_block`, `selective_ssm`, `simple_ssm`, `svd_mamba`):** Map $[W_V; W_Q]$ into the primary input projections of `in_proj` and map $W_O$ into `out_proj`, preserving the principal semantic subspace on step 0.

### 3. Instant Closed-Form Layerwise Ridge Alignment in Stage 0 (`calibrate_surgical_model`)
- Collect layer input activations $X_\ell$ across 16 calibration sequences (taking $< 2.5$ seconds on GPU).
- Let $Z_\ell \in \mathbb{R}^{N \times D}$ be the output of the new attention module, and $Y_\ell \in \mathbb{R}^{N \times D}$ be the output of the teacher's original attention.
- Solve the Tikhonov-regularized Ridge problem in closed form:
  $$W_{\text{align}}^{(\ell)} = \left( Z_\ell^\top Z_\ell + \lambda I \right)^{-1} Z_\ell^\top Y_\ell, \quad \lambda = 10^{-2}$$
- Fold $W_{\text{align}}^{(\ell)}$ directly into `out_proj`:
  $$W_{\text{out, calibrated}}^{(\ell)} \leftarrow W_{\text{align}}^{(\ell)} \cdot W_{\text{out}}^{(\ell)}$$

### 4. Self-Contained Hugging Face `AutoModel` Export & Reload
- Create `neuralops/export/hf_hub_surgery.py` with `save_surgical_bert_for_hf_hub`:
  1. Saves model weights, tokenizer, `training_metrics.json`, and Model Card `README.md`.
  2. Embeds `"neuralops_surgery": {"attn_key": attn_key, "embed_dim": 768, "num_heads": 12}` and `"auto_map"` into `config.json`.
  3. Bundles self-contained `modeling_neuralops_bert.py` so any user can load the model via `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`.

### 5. Deterministic 15% Masked LM Perplexity & Fast Tensor-Core Pillar 4
- Evaluate Perplexity using **deterministic 15% Masked Language Modeling** (`seed=42`, `labels = -100` on unmasked tokens):
  $$\text{MLM-PPL} = \exp\left( \min\left( \mathcal{L}_{\text{MLM}}, 20.0 \right) \right)$$
- Optimize Pillar 4 (SVD Spectrum Loss) by computing **Fast Normalized Gram Covariance Matching** in FP16 on Tensor Cores:
  $$\mathcal{L}_{\text{Cov}} = \frac{1}{D^2} \left\| \frac{h_S^\top h_S}{\|h_S\|_F} - \frac{h_T^\top h_T}{\|h_T\|_F} \right\|_F^2$$
- In `upload_to_hf`, enforce `--cleanup_weights_after_upload 1`, deleting local `.safetensors` files upon confirmed commit hash to maintain active disk usage under **2 GB**.

---

## 3. The Dual-Track Architecture: BERT Encoder Zoo & Qwen Causal LM Roadmap

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              NEURALOPS DUAL-TRACK ARCHITECTURE                         │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ TRACK A: Encoder Foundation (Immediate)   │ TRACK B: Modern Generative LM (Expansion)  │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Model: bert-base-uncased (110M params)  │ • Model: Qwen/Qwen2.5-0.5B / SmolLM2-135M  │
│ • Topology: 12 Layers, 768 Dim, 12 Heads  │ • Topology: 24 Layers, 896 Dim, GQA 14:2   │
│ • Task: Masked LM & Sentence Embeddings   │ • Task: Autoregressive Text Gen & Chat     │
│ • Dataset: WikiText-2 (seq_len = 128)     │ • Dataset: WikiText-103 & SST-2 Downstream │
│ • State Complexity: Standard O(S)         │ • State Complexity: Constant O(1) Memory!  │
│ • Scope: All 131 Canonical Mechanisms     │ • Scope: Top-20 High-Speed Linear Champions│
│ • Target Hub Org: Hooshaai                │ • Target Hub Org: Hooshaai                 │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

Track A provides the comprehensive empirical foundation: all 131 attention mechanisms tested under strictly controlled conditions. Track B takes the Pareto-optimal linear and subquadratic champions from Track A and applies them to `Qwen2.5-0.5B`, eliminating the quadratic KV-cache bottleneck for long-context generation ($O(1)$ memory).

---

## 4. Mathematical Formulation: MLM Pseudo-Perplexity & Metric Integrity

For bidirectional encoder models like BERT, unmasked evaluation leaks the target token into self-attention, collapsing loss to zero.

### Mathematical Formulation
1. **Deterministic Masking:** For validation sequence $X = (x_1, \dots, x_S)$ with $S=128$, generator seed `seed = 42 + batch_idx` selects 15% of token positions $\mathcal{M} \subset \{1, \dots, S\}$ (excluding `[CLS]` and `[SEP]`).
2. **80/10/10 Token Distribution:**
   - 80% of tokens in $\mathcal{M}$ become `[MASK]` (token ID 103).
   - 10% become a uniform random token $v \in \mathcal{V}$.
   - 10% remain unchanged.
3. **Loss Function:** Unmasked tokens receive target $-100$. Loss is strictly the cross-entropy over masked positions:
   $$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P_{\text{model}}(x_i \mid \tilde{X})$$
4. **Metric Equations:**
   $$\text{MLM-PPL} = \exp\left( \min\left( \mathcal{L}_{\text{MLM}}, 20.0 \right) \right)$$
   $$\text{Masked Token Accuracy (\%)} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathbb{I}\left( \arg\max_{v \in \mathcal{V}} z_i(v) = x_i \right) \times 100\%$$

---

## 5. The 14-Phase Empirical Screening & Distillation Funnel

### Phase 0: Hardware, Invariants & Registry Inventory Audit
- **Objective:** Establish physical silicon baseline on Sharif HPC (`GPU-3090-1`), verify PyTorch/CUDA environment isolation, and dynamically index all attention mechanisms.
- **Code Reuse:** `neuralops/core/registry.py` (`ATTENTION_REGISTRY`), `neuralops/trainer/smart_gpu_trainer.py` (`GPUMemoryManager.get_device_info`).
- **Actions:**
  1. Print physical GPU capabilities (RTX 3090, SM 8.6, 24GB GDDR6X, CUDA 11.3).
  2. Dynamically enumerate all 115 registered keys in `ATTENTION_REGISTRY` + 16 extensible units (131 total).
  3. Validate that no zombie jobs exist: `pgrep -af 'pytest|neuralops|triton'` and `qstat -u $USER`.

---

### Phase 1: Core Surgery Engine Upgrade & Single-$W_O$ Contract Enforcement
- **Objective:** Implement the 5 critical surgery fixes in `neuralops/core/` and `neuralops/layers/bert_adapter.py`.
- **Code Reuse:** `_attention_injectors.py`, `_attention_wrapper.py`, `_attention_surgery.py`, `_attention_helpers.py`.
- **Actions:**
  1. Update `_wrap_and_replace_layer` to extract $(W_O, b_O)$ from `layer.attention.output.dense`, fold it into `new_attn.out_proj`, and replace `layer.attention.output.dense` with `nn.Identity()`.
  2. Enforce `is_causal=False` in `UniversalAttentionWrapper` for encoder backbones.
  3. Implement family-specific weight initialization (forget-gate bias $b_g \in [2.0, 3.0]$ for gated models, truncated SVD for low-rank models, in-proj slice mapping for SSMs).
  4. Implement `calibrate_surgical_model` in `_attention_surgery.py`.
- **Verification Gate:** Run `tests/test_bert_surgery_and_zoo_pipeline.py`.

---

### Phase 2: Teacher Baseline Telemetry (BERT 15% MLM PPL & Qwen CLM PPL)
- **Objective:** Establish canonical reference metrics on validation sets before surgery.
- **Code Reuse:** `neuralops/evaluation/eval_perplexity.py` (`build_calibration_loader`), `scripts/benchmarks/bert_zoo_baseline_ppl.py`.
- **Actions:**
  1. Pre-tokenize `wikitext-2-raw-v1` into $S=128$ chunks.
  2. Evaluate unmodified `bert-base-uncased` under deterministic 15% MLM masking (`seed=42`).
  3. Record `ppl_vanilla_bert` and `acc_vanilla_bert` to `results/bert_attention_zoo/baseline_ppl.json`.
  4. Evaluate unmodified `Qwen2.5-0.5B` on WikiText-103; record causal perplexity to `results/qwen_attention_zoo/baseline_clm_ppl.json`.

---

### Phase 3: GPU Zero-Shot Instantiation & Stability Screening (All 131 Modules)
- **Objective:** Test every registered attention mechanism on physical GPU under BERT and Qwen architectures to classify instantiability and numerical stability.
- **Code Reuse:** `scripts/benchmarks/bert_zoo_discover_keys.py`, `neuralops/evaluation/eval_patcher.py` (`build_model_instance`).
- **Actions:**
  For each key $k \in \text{ATTENTION\_REGISTRY}$:
  1. Instantiate module with `embed_dim=768, num_heads=12`.
  2. Perform forward/backward step on GPU with random input tensor; verify gradient norms are finite (no `NaN`/`Inf`).
  3. Graft onto `bert-base-uncased` via `replace_attention(model, attention=k, copy_weights=True)`.
  4. Categorize each module:
     - `STATUS_STABLE`: Finite loss and gradients on step 0.
     - `STATUS_RECOVERABLE`: Finite after Stage-0 Ridge calibration.
     - `STATUS_INCOMPATIBLE`: Structural failure (logged to `skipped_keys.json` with stack trace).
  5. Write valid keys to `results/bert_attention_zoo/registry_keys.json`.

---

### Phase 4: Stage 0 Closed-Form Ridge Calibration Profiling & Metric Delta Recording
- **Objective:** Quantify the exact error reduction provided by closed-form Ridge alignment before gradient backpropagation.
- **Code Reuse:** `neuralops/core/_attention_surgery.py` (`calibrate_surgically_replaced_model`).
- **Actions:**
  1. For each module from Phase 3, record `raw_surgery_ppl` immediately after grafting.
  2. Run `calibrate_surgical_model` using 16 calibration batches ($< 3.0$ seconds on RTX 3090).
  3. Record `calibrated_stage0_ppl` and compute alignment delta:
     $$\Delta \text{PPL}_{\text{Stage 0}} = \text{raw\_surgery\_ppl} - \text{calibrated\_stage0\_ppl}$$
  4. Confirm that $\Delta \text{PPL}_{\text{Stage 0}} \ge 0$ across all modules.

---

### Phase 5: 60-Second Pilot KD Screening & Multi-Objective Pareto Frontier Analysis
- **Objective:** Execute a fast 60-second pilot distillation on GPU for each module to analyze convergence trajectories and multi-objective trade-offs.
- **Code Reuse:** `neuralops/trainer/distillation/multi_level_kd.py` (`MultiLevelKDTrainer`), `neuralops/trainer/smart_gpu_trainer.py` (`AsyncDataPrefetcher`).
- **Multi-Objective Pareto Metrics:**
  1. **Perplexity Delta:** $\Delta \text{PPL}_{60\text{s}} = \text{calibrated\_stage0\_ppl} - \text{post\_60s\_ppl}$.
  2. **Throughput:** Tokens processed per second ($\text{tokens/sec}$) on RTX 3090.
  3. **VRAM Footprint:** Peak allocated megabytes during training.
  4. **Subspace Similarity:** Centered Kernel Alignment ($\text{CKA}$) similarity against teacher layer outputs.
- **Output:** Generate Pareto scatter plot and ranking ledger in `results/bert_attention_zoo/pilot_pareto_ranking.json`.

---

### Phase 6: Attention Mechanism Selection & Classification
- **Objective:** Classify all 131 attention mechanisms based on Phase 5 Pareto metrics into distinct operational tiers.
- **Classification Tiers:**
  - **Tier-1 Champions (Top 20%):** Pareto-optimal trade-off between throughput, low VRAM, and rapid PPL recovery (e.g. `gated_delta_net`, `gla_attention`, `cosformer`, `svd_linear_attention`, `flash_attention3`). Prioritized for full 5-minute training and Track B Qwen generative scaling.
  - **Tier-2 Standard (Remaining Valid Modules):** Scheduled for complete 5-minute distillation in the cluster batch queue.
  - **Tier-3 Quarantined:** Modules exhibiting numerical instability or excessive memory consumption, slated for isolated mathematical postmortem.

---

### Phase 7: Full 5-Minute (300s) 3-Stage 6-Pillar KD Distillation on Selected Modules
- **Objective:** Execute production 300.0-second knowledge distillation on each selected module with 100% RTX 3090 GPU saturation.
- **Code Reuse:** `scripts/benchmarks/bert_zoo_train_wiki.py`, `scripts/benchmarks/run_layerwise_cached_kd_ablation.py`.
- **Curriculum Stages:**
  - **Stage 0 ($t = 0\dots 5\,\text{s}$):** Instant Ridge output projection alignment. Record `pre_kd_ppl` and `pre_kd_acc`.
  - **Stage 1 ($t = 5\dots 95\,\text{s}$, 30%):** Frozen-backbone subspace distillation (AdamW $\text{lr}=1.0 \times 10^{-3}$) optimizing Hidden MSE + Linear CKA + Fast Gram Covariance.
  - **Stage 2 ($t = 95\dots 300\,\text{s}$, 70%):** Full-model unfrozen KD with differential learning rates ($\text{lr}_{\text{attn}}=5.0 \times 10^{-4}, \text{lr}_{\text{backbone}}=5.0 \times 10^{-5}$) optimizing Fused Logits KL + MLM Cross-Entropy.
  - **Post-Eval ($t \approx 305\,\text{s}$):** Record `post_kd_ppl`, `post_kd_acc`, throughput, and peak VRAM.
- **Dynamic VRAM Saturation:** Starts at `batch_size=128`, `grad_accum=2` (utilizing 18–20 GB VRAM). Halves to 64 upon OOM exception.

---

### Phase 8: Asynchronous Background HF Hub Upload & Storage Quota Protection (`Hooshaai`)
- **Objective:** Stream completed checkpoints to Hugging Face Hub under `Hooshaai` without halting GPU training, while keeping active disk usage below 2 GB.
- **Code Reuse:** `neuralops/hub/uploader.py` (`AsyncHFHubUploader`), `scripts/hpc/bert_zoo_upload_hf.py`.
- **Protocol:**
  1. GPU worker finishes Module $i$, writes checkpoint to staging directory, and pushes event to `AsyncHFHubUploader`.
  2. GPU immediately begins training Module $i+1$ with zero idle time.
  3. Background uploader daemon:
     - Creates repository `Hooshaai/bert-{slug}-wiki`.
     - Generates publication-grade Model Card `README.md`.
     - Calls `HfApi.upload_folder(...)`.
     - Confirms remote commit hash.
     - Writes `upload_success.json`.
     - **Deletes local `model.safetensors` / `pytorch_model.bin` to protect disk quota.**
  4. Master ledger (`metrics.jsonl`) updated atomically.

---

### Phase 9: Self-Contained Zero-Dependency Checkpoint Validation (`auto_map`)
- **Objective:** Verify that any uploaded checkpoint can be downloaded and used by third-party researchers without installing NeuralOps from source.
- **Code Reuse:** `neuralops/export/hf_hub_surgery.py` (`save_surgical_bert_for_hf_hub`), `neuralops/models/hub.py` (`push_to_hub`).
- **Protocol:**
  1. Each repository bundles `modeling_neuralops_bert.py` and patched `config.json` with `auto_map`:
     ```json
     {
       "auto_map": {
         "AutoConfig": "modeling_neuralops_bert.NeuralOpsBertConfig",
         "AutoModelForMaskedLM": "modeling_neuralops_bert.NeuralOpsBertForMaskedLM"
       }
     }
     ```
  2. Execute isolated test:
     ```python
     from transformers import AutoModelForMaskedLM, AutoTokenizer
     model = AutoModelForMaskedLM.from_pretrained("Hooshaai/bert-gated-delta-net-wiki", trust_remote_code=True)
     ```
  3. Verify that masked token prediction succeeds with zero missing-key warnings.

---

### Phase 10: Track B — Qwen2.5-0.5B Generative Causal LM Distillation Sweep (Constant $O(1)$ KV)
- **Objective:** Expand the zoo to modern generative text models by grafting the Tier-1 linear attention champions into `Qwen/Qwen2.5-0.5B`.
- **Code Reuse:** `neuralops/core/model_family.py` (`ModelAdapter`), `tests/gpu_exhaustive/test_surgery_qwen_gqa.py`.
- **Protocol:**
  1. Graft linear attention champions (`gated_delta_net`, `gla_attention`, `cosformer`) into `Qwen2.5-0.5B` matching GQA 14:2 head configuration.
  2. Distill on `WikiText-103` using 3-Stage Causal KD (Teacher Causal Logits KL + Autoregressive Next-Token Cross-Entropy).
  3. Measure KV-cache memory footprint across sequence lengths $S \in [1024, 4096, 16384, 32768]$; confirm that memory remains **strictly constant $O(1)$**.
  4. Upload checkpoints to `Hooshaai/qwen-2.5-0.5b-{slug}-wiki`.

---

### Phase 11: Multi-Format Packaging for Universal Accessibility (GGUF, Ollama, vLLM / SGLang)
- **Objective:** Enable universal accessibility across edge laptops, CPUs, and high-throughput production servers.
- **Code Reuse:** `neuralops/export/gguf_exporter.py`, `neuralops/integrations/vllm.py`, `neuralops/integrations/sglang.py`.
- **Outputs:**
  1. **GGUF / Ollama:** Convert distilled checkpoints via `gguf_exporter.py` for instant zero-GPU execution:
     ```bash
     ollama run hooshaai/qwen-gla:0.5b
     ```
  2. **vLLM Integration:** Connect linear recurrent states with `vLLM` custom runner for production serving:
     ```bash
     vllm serve Hooshaai/qwen-2.5-0.5b-gla-wiki --port 8000
     ```

---

### Phase 12: Interactive Gradio Attention Arena (`Hooshaai/bert-attention-zoo-demo`)
- **Objective:** Deploy a live demonstration space on Hugging Face Spaces (`Hooshaai/bert-attention-zoo-demo`).
- **Code Reuse:** `spaces/bert_attention_demo/app.py`.
- **Features:**
  - Interactive dropdown for all 131 attention mechanisms.
  - Real-time `[MASK]` prediction probe and top-5 token probabilities.
  - Side-by-side performance cards: Vanilla Teacher PPL vs Zero-Shot PPL vs Post-KD PPL.
  - 1-click copy-paste Python snippet for downstream fine-tuning.

---

### Phase 13: End-User API & CLI Integration
- **Objective:** Expose clean, 1-line client interfaces in `neuralops`.
- **Code Reuse:** `neuralops/hub/bert_zoo.py`, `neuralops/benchmarks/cli.py`.
- **Client Interfaces:**
  ```python
  import neuralops

  # 1-Line download and instantiation
  model, tokenizer = neuralops.load_pretrained_model(
      model_name="bert-base-uncased",
      attention_type="gated_delta_net",
      task="masked-lm",
  )
  ```
  ```bash
  # CLI commands
  neuralops hub list
  neuralops hub pull --attention gated_delta_net
  neuralops hub train-zoo --minutes-per-module 5.0 --upload-hf --hf-org Hooshaai
  ```

---

### Phase 14: Master Benchmark Ledger Synchronization, Paper Tables & CI Release Gates
- **Objective:** Persist all empirical measurements into the master results ledger and generate publication artifacts.
- **Code Reuse:** `neuralops/ui/queries/ledger.py`, `scripts/export_paper_tables.py`.
- **Artifacts:**
  - `results/bert_attention_zoo/master_ledger.sqlite`
  - `results/bert_attention_zoo/metrics.jsonl`
  - `results/bert_attention_zoo/summary_table.csv`
  - Publication-ready LaTeX tables ranking attention mechanisms by parameter efficiency, throughput, and perplexity recovery.

---

## 6. Catalog of All 131 Canonical Attention Units

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

## 7. Hardware & VRAM Saturation Budget on RTX 3090

### VRAM Allocation (24,576 MB Total GDDR6X)

| Component | Allocation | Role |
|---|---|---|
| **Cached Teacher Representations** | ~2,500 MB | Pre-computed layer outputs & logits in FP16 (eliminates 50% FLOPs) |
| **Student BERT Weights** | ~440 MB | Active parameters undergoing distillation |
| **AdamW Optimizer States** | ~880 MB | FP32 momentum and variance buffers |
| **FP32 Gradients** | ~440 MB | Accumulated gradient buffers |
| **Pre-Tokenized Dataset Tensor** | ~60 MB | Full WikiText-2 token chunks resident in memory |
| **Activation Cache ($B=128, S=128$)** | ~14,000 MB | 12 transformer layers under mixed-precision backprop |
| **Total Peak Utilization** | **~18,320 MB** | **~75% to 80% VRAM Saturation** |
| **Headroom Buffer** | **~6,256 MB** | Safety margin preventing OOM on modules with state matrices |

---

## 8. Failure Policies, Recovery Matrix & Verification Protocols

| Fault Condition | Automated Handling & Self-Healing |
|---|---|
| **OOM on Attention $i$** | Intercept `torch.cuda.OutOfMemoryError`, empty VRAM cache, halve batch size ($128 \rightarrow 64$), double accumulation ($2 \rightarrow 4$), retry once. |
| **HPC Outbound Network Blocked** | PBS node stages weights locally; login-node watcher daemon detects `training_metrics.json` and performs upload asynchronously. |
| **Walltime Preemption** | Resume capability: engine scans `training_metrics.json` and skips previously completed checkpoints upon re-launch. |
| **Divergent Loss / NaN** | Flagged as `FAIL_NUMERICS` in `metrics.jsonl`; process moves cleanly to module $i+1$. |
| **Disk Quota Constraint** | Local `.safetensors` deleted immediately upon upload commit verification; active disk usage $< 2\text{ GB}$. |

---

## Master Checklist

- [x] **Master Architectural Plan Documented:** `docs/MASTER_DUAL_ZOO_AND_SURGERY_PLAN.md`
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

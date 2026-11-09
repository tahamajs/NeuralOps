# 🏛️ NeuralOps Master Dual-Zoo & Empirical Selection Plan
## Multi-Phase GPU Screening, Surgery Upgrades, Pareto Selection & End-to-End Distillation for BERT-131 & Qwen-Generative Zoos

**Document ID:** `NOP-MASTER-ZOO-SELECTION-v4.0`
**Status:** Authoritative Multi-Phase Empirical Roadmap
**Target Architectures:**
  - **Track A (NLU & Encoder Zoo):** `bert-base-uncased` (110M params, 12 layers, 768-dim, 12 heads)
  - **Track B (Generative & Causal LM Zoo):** `Qwen/Qwen2.5-0.5B` (490M params, 24 layers, GQA 14:2 heads) & `SmolLM2-135M`
**Primary Compute Host:** Sharif HPC Cluster (`GPU-3090-1`, NVIDIA RTX 3090 24GB GDDR6X, CUDA 11.3, GCC 8.2)
**Distributed Secondary Nodes:** `ssh gpu` (RTX 4070 8GB SM 8.9) & `ssh tahoora` (GTX 1060 6GB SM 6.1)
**Hugging Face Hub Organization:** `Hooshaai` (Official Hub Org)
**Security & Credential Mandate:** Absolute redaction of `$HF_TOKEN` — passed via environment variables only

---

## Executive Summary

This master plan establishes a rigorous, **empirical-first multi-phase protocol** to evaluate, benchmark, select, distill, and publish the **NeuralOps Pretrained Attention Zoo** on Hugging Face Hub under **`Hooshaai`**.

Rather than blindly running 5-minute training sweeps on untested modules, this plan incorporates a **staged screening funnel**:
1. **Phase 1–2:** Architectural upgrades to the Model Surgery Engine (fixing the Double-$W_O$ bug, family-specific smart weight initialization, Stage-0 Ridge calibration).
2. **Phase 3–5:** Progressive GPU screening across all 131 registered attention mechanisms (Instantiability $\rightarrow$ Zero-Shot Stability $\rightarrow$ Stage-0 Ridge Gain $\rightarrow$ 60-second Pilot KD Pareto ranking).
3. **Phase 6–8:** Full 5-minute (300s) 100% GPU-saturated 3-Stage KD on the validated attention inventory, combined with non-blocking asynchronous Hugging Face Hub uploads and local disk quota protection ($< 2\text{ GB}$ active footprint).
4. **Phase 9–12:** Generative Causal LM distillation (`Qwen2.5-0.5B` with $O(1)$ KV-cache linear attention), multi-format export (GGUF for Ollama, vLLM serving), and the interactive Gradio Attention Arena (`Hooshaai/bert-attention-zoo-demo`).

---

## Complete Multi-Phase Roadmap Funnel

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE NEURALOPS EMPIRICAL SELECTION FUNNEL                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 0: Hardware, Invariants & Registry Audit (131 Units, RTX 3090 Telemetry)         │
│                                    ▼                                                   │
│ PHASE 1: Core Surgery Engine Upgrades (Single-W_O Contract, Smart Init, Ridge Calib)   │
│                                    ▼                                                   │
│ PHASE 2: Baseline Telemetry (Vanilla BERT 15% MLM PPL & Qwen2.5 Causal Baseline)       │
│                                    ▼                                                   │
│ PHASE 3: Zero-Shot Surgery & Stability Screening (All 131 Modules on Physical GPU)    │
│          ├── Passed (Stable Gradient & Output) ──────────────────────────┐             │
│          └── Failed/Diverged ──► Quarantined & Filtered                  │             │
│                                                                          ▼             │
│ PHASE 4: Stage 0 Closed-Form Ridge Alignment Benchmark (16 Calibration Batches)        │
│          └── Solves W_align = (Z^T Z + λI)^-1 Z^T Y in FP32              │             │
│                                                                          ▼             │
│ PHASE 5: Short-Horizon (60s) Pilot KD Screening & Pareto Multi-Objective Ranking       │
│          └── Selects Tier-1 Champions & Confirms Convergence Trajectories               │
│                                                                          ▼             │
│ PHASE 6: Production 5-Minute (300s) 3-Stage KD Distillation (100% VRAM Saturation)     │
│          ├── Stage 0: Ridge Output Fold (0-5s)                                         │
│          ├── Stage 1: Frozen Subspace CKA/MSE Mimicry (5-95s)                           │
│          └── Stage 2: End-to-End Differential-LR Polish (95-300s)                      │
│                                                                          ▼             │
│ PHASE 7: Asynchronous Non-Blocking HF Upload & Storage Cleanup (Hooshaai Org)          │
│          └── Confirms Commit Hash ──► Deletes local .safetensors (Disk < 2 GB)         │
│                                                                          ▼             │
│ PHASE 8: Self-Contained AutoModel Export Validation (auto_map + Zero-Dependency Pipe)  │
│                                    ▼                                                   │
│ PHASE 9: Track B — Qwen2.5-0.5B Generative Causal LM Distillation Sweep (O(1) KV)      │
│                                    ▼                                                   │
│ PHASE 10: Multi-Format Export (GGUF for Ollama, ONNX Runtime, vLLM / SGLang Serving)   │
│                                    ▼                                                   │
│ PHASE 11: Interactive Gradio Model Arena (Hooshaai/bert-attention-zoo-demo Space)      │
│                                    ▼                                                   │
│ PHASE 12: Master Benchmark Ledger Synchronization, Paper Tables & Release Gate         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase Specifications

### Phase 0: Hardware, Invariants & Registry Audit

- **Objective:** Establish physical silicon baseline on Sharif HPC (`GPU-3090-1`), verify PyTorch/CUDA environment isolation, and dynamically index all attention mechanisms.
- **Actions:**
  1. Verify GPU device name, compute capability (SM 8.6), VRAM capacity (24,576 MB GDDR6X), driver (470+ / 535+), and CUDA compiler (`nvcc` 11.3).
  2. Dynamically enumerate `ATTENTION_REGISTRY` and `COMPRESSOR_REGISTRY` in `neuralops/core/registry.py`.
  3. Confirm that no stale processes (`pytest`, `python`, `triton`) are running on the allocated GPU node.
- **Verification Gate:**
  ```bash
  python3 -c "import torch; assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 23e9"
  ```

---

### Phase 1: Core Surgery Engine Upgrades

- **Objective:** Upgrade `neuralops/core/` to resolve the 5 architectural flaws and prevent zero-shot representation corruption.
- **Key Deliverables:**
  1. **Single-$W_O$ Output Projection Contract (`_attention_injectors.py`, `_attention_wrapper.py`):**
     - When injecting into BERT/RoBERTa `layer.attention.self`, extract $(W_Q, b_Q, W_K, b_K, W_V, b_V)$ from `layer.attention.self` and $(W_O, b_O)$ from `layer.attention.output.dense`.
     - Transplant into the new module (`q_proj, k_proj, v_proj, out_proj`).
     - Replace `layer.attention.output.dense` with `nn.Identity()` to guarantee strictly one output projection per layer.
     - Enforce `is_causal=False` in `UniversalAttentionWrapper` for bidirectional encoder models.
  2. **Family-Specific Smart Weight Initialization:**
     - *Gated Linear Attention (`gated_delta_net`, `gla_attention`, `kda_attention`):* Initialize forget-gate bias $b_g \in [2.0, 3.0]$ such that initial retention $\sigma(b_g) \ge 0.88$.
     - *SVD / Low-Rank Modules (`svd_linear_attention`, `asvd_attention`):* Perform truncated SVD on pretrained $W_Q, W_K, W_V, W_O$, initializing $U_r = U_{:, :r}\sqrt{\Sigma_{:r}}$ and $V_r = \sqrt{\Sigma_{:r}}V_{:r, :}^\top$.
     - *SSM / Mamba Modules (`mamba_block`, `simple_ssm`, `svd_mamba`):* Map $[W_V; W_Q]$ into `in_proj` and $W_O$ into `out_proj`.
  3. **Instant Closed-Form Ridge Calibration Function:**
     - Implement `calibrate_surgical_model(teacher, student, calibration_loader, ridge_lambda=1e-2)` in `neuralops/core/_attention_surgery.py`.
- **Verification Gate:** Unit test in `tests/test_bert_surgery_and_zoo_pipeline.py` confirming `layer.attention.output.dense` is `nn.Identity()` and forward shapes match $[B, S, D]$.

---

### Phase 2: Baseline Telemetry & Dataset Preparation

- **Objective:** Establish canonical teacher baseline metrics on validation sets with zero data leakage.
- **Key Deliverables:**
  1. **WikiText-2 MLM Baseline (`bert-base-uncased`):**
     - Pre-tokenize `wikitext-2-raw-v1` into contiguous $S=128$ chunks.
     - Apply deterministic 15% masking (80% `[MASK]`, 10% random token, 10% identity) with generator `seed=42`.
     - Evaluate unmodified teacher: compute $\text{PPL}_{\text{MLM}} = \exp(\mathcal{L}_{\text{MLM}})$ and $\text{Acc}_{\text{MLM}}$ (% masked tokens correctly predicted).
     - Save reference to `results/bert_attention_zoo/baseline_ppl.json`.
  2. **WikiText-103 / SST-2 Causal Baseline (`Qwen2.5-0.5B`):**
     - Measure autoregressive causal cross-entropy loss, causal perplexity $\exp(\mathcal{L}_{\text{CLM}})$, and SST-2 zero-shot accuracy.
     - Save reference to `results/qwen_attention_zoo/baseline_clm_ppl.json`.
- **Verification Gate:** Output files exist with finite, non-zero floats (e.g. BERT MLM-PPL $\approx 6.84$, Acc $\approx 68.4\%$).

---

### Phase 3: Zero-Shot Surgery & Stability Screening (All 131 Modules)

- **Objective:** Test every registered attention mechanism on physical GPU under BERT and Qwen architectures to classify instantiability and numerical stability.
- **Protocol:**
  For each key $k \in \text{ATTENTION\_REGISTRY}$:
  1. Instantiate via `create_attention(k, embed_dim=768, num_heads=12)`.
  2. Wrap with `BertAttentionAdapter` and perform dummy forward/backward pass with random tensor.
  3. Perform full surgical graft onto `bert-base-uncased` via `replace_attention(model, attention=k, copy_weights=True)`.
  4. Run forward pass on 1 validation batch; check for `NaN`, `Inf`, or gradient explosion.
  5. Classify into:
     - `STATUS_STABLE`: Finite loss, correct output shape $[B, S, 768]$.
     - `STATUS_RECOVERABLE`: Finite after Stage-0 Ridge calibration.
     - `STATUS_INCOMPATIBLE`: Structural failure (logged to `skipped_keys.json` with stack trace).
- **Deliverables:** `results/bert_attention_zoo/registry_keys.json` containing only verified stable/recoverable keys.

---

### Phase 4: Stage 0 Closed-Form Ridge Calibration Benchmark

- **Objective:** Quantify the exact error reduction provided by closed-form Ridge alignment before gradient backpropagation.
- **Protocol:**
  1. For each verified module from Phase 3, record `raw_surgery_ppl` immediately after grafting.
  2. Run `calibrate_surgical_model` using 16 calibration batches from WikiText-2 ($< 3.0$ seconds on RTX 3090).
  3. Record `calibrated_stage0_ppl` and compute alignment delta:
     $$\Delta \text{PPL}_{\text{Stage 0}} = \text{raw\_surgery\_ppl} - \text{calibrated\_stage0\_ppl}$$
  4. Confirm that $\Delta \text{PPL}_{\text{Stage 0}} \ge 0$ across all modules (verifying that Ridge calibration never degrades representation quality).

---

### Phase 5: Short-Horizon (60s) Pilot KD Screening & Pareto Ranking

- **Objective:** Run a fast 60-second pilot distillation to rank and select the highest-performing attention mechanisms across architectural families before the full 5-minute investment.
- **Multi-Objective Pareto Criteria:**
  1. **Convergence Speed:** $\Delta \text{PPL}_{60\text{s}} = \text{calibrated\_stage0\_ppl} - \text{post\_60s\_ppl}$.
  2. **Computational Throughput:** Tokens processed per second ($\text{tok/s}$) on RTX 3090.
  3. **VRAM Footprint:** Peak allocated megabytes during forward/backward pass.
  4. **Subspace Fidelity:** Centered Kernel Alignment ($\text{CKA}$) similarity against teacher activations.
- **Selection Funnel Output:**
  - **Tier-1 Champions (Top 25%):** Prioritized for immediate full 5-minute distillation and Track B Qwen generative scaling.
  - **Tier-2 Standard (Remaining Valid Modules):** Scheduled for complete batch distillation.
  - **Tier-3 Unstable:** Quarantined for isolated mathematical postmortem.

---

### Phase 6: Production 5-Minute (300s) 3-Stage KD Distillation

- **Objective:** Execute the comprehensive 300.0-second knowledge distillation on each selected module with 100% RTX 3090 GPU saturation.
- **Curriculum Execution:**
  - **Stage 0 ($t = 0\dots 5\,\text{s}$):** Instant Ridge output projection alignment.
  - **Pre-Eval ($t \approx 5\,\text{s}$):** Record `pre_kd_ppl` and `pre_kd_acc`.
  - **Stage 1 ($t = 5\dots 95\,\text{s}$, 30%):** Frozen-backbone subspace distillation (AdamW $\text{lr}=1.0 \times 10^{-3}$) optimizing Hidden MSE + Linear CKA + Fast Gram Covariance.
  - **Stage 2 ($t = 95\dots 300\,\text{s}$, 70%):** Full-model unfrozen KD with differential learning rates ($\text{lr}_{\text{attn}}=5.0 \times 10^{-4}, \text{lr}_{\text{backbone}}=5.0 \times 10^{-5}$) optimizing Fused Logits KL + MLM Cross-Entropy.
  - **Post-Eval ($t \approx 305\,\text{s}$):** Record `post_kd_ppl`, `post_kd_acc`, throughput, and peak VRAM.
- **Dynamic VRAM Saturation:** Starts at `batch_size=128`, `grad_accum=2` (utilizing 18–20 GB VRAM). Halves to 64 upon OOM exception.

---

### Phase 7: Asynchronous Non-Blocking HF Upload & Storage Protection

- **Objective:** Stream completed checkpoints to Hugging Face Hub under `Hooshaai` without halting GPU training, while keeping active disk usage below 2 GB.
- **Pipeline:**
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

### Phase 8: Self-Contained AutoModel Export Validation

- **Objective:** Verify that any uploaded checkpoint can be downloaded and used by third-party researchers without installing NeuralOps from source.
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

### Phase 9: Track B — Qwen2.5-0.5B Generative Causal LM Distillation

- **Objective:** Expand the zoo to modern generative text models by grafting the Tier-1 linear attention champions into `Qwen/Qwen2.5-0.5B`.
- **Key Engineering Milestones:**
  1. Implement `neuralops/layers/qwen_adapter.py` mapping GQA 14:2 head topology to linear recurrent state matrices.
  2. Replace standard softmax attention with $O(1)$ state recurrent kernels (`gated_delta_net`, `gla_attention`, `cosformer`).
  3. Distill on `WikiText-103` using 3-Stage Causal KD (Teacher Causal Logits KL + Autoregressive Next-Token Cross-Entropy).
  4. Verify that KV-cache memory remains **strictly constant $O(1)$** across prompt lengths from 1,024 to 32,768 tokens.
  5. Upload to `Hooshaai/qwen-2.5-0.5b-{slug}-wiki`.

---

### Phase 10: Multi-Format Packaging (GGUF, Ollama, vLLM)

- **Objective:** Enable universal accessibility across edge laptops, CPUs, and high-throughput production servers.
- **Formats:**
  1. **GGUF / Ollama:** Convert distilled checkpoints via `neuralops/export/gguf_exporter.py` for instant zero-GPU execution:
     ```bash
     ollama run hooshaai/qwen-gla:0.5b
     ```
  2. **vLLM Integration:** Connect linear recurrent states with `vLLM` custom runner for production serving:
     ```bash
     vllm serve Hooshaai/qwen-2.5-0.5b-gla-wiki --port 8000
     ```

---

### Phase 11: Interactive Gradio Model Arena

- **Objective:** Deploy a live demonstration space on Hugging Face Spaces (`Hooshaai/bert-attention-zoo-demo`).
- **Features:**
  - Interactive dropdown for all 131 attention mechanisms.
  - Real-time `[MASK]` prediction probe and top-5 token probabilities.
  - Side-by-side performance cards: Vanilla Teacher PPL vs Zero-Shot PPL vs Post-KD PPL.
  - 1-click copy-paste Python snippet for downstream fine-tuning.

---

### Phase 12: Master Benchmark Ledger Synchronization & Paper Tables

- **Objective:** Persist all empirical measurements into the master results ledger and generate LaTeX paper tables.
- **Persistence Artifacts:**
  - `results/bert_attention_zoo/master_ledger.sqlite`
  - `results/bert_attention_zoo/metrics.jsonl`
  - `results/bert_attention_zoo/summary_table.csv`
  - Publication LaTeX table ranking attention mechanisms by parameter efficiency, throughput, and perplexity recovery.

---

## Complete 131 Attention Registry Catalog

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

## File Implementation & Verification Matrix

| Subsystem File | Purpose & Architectural Role | Phase |
|---|---|:---:|
| `neuralops/layers/bert_adapter.py` | Universal BERT adapter with single-$W_O$ output contract | Phase 1 |
| `neuralops/core/_attention_surgery.py` | Closed-form Stage-0 Ridge alignment & smart weight initialization | Phase 1 |
| `neuralops/hub/uploader.py` | Asynchronous HF Hub uploader daemon with automated weight cleanup | Phase 7 |
| `neuralops/hub/bert_zoo.py` | High-level client API: `load_pretrained_model`, `list_pretrained_models` | Phase 8 |
| `scripts/benchmarks/bert_zoo_discover_keys.py` | Phase 0 & 3 dynamic key discovery and forward probe | Phase 3 |
| `scripts/benchmarks/bert_zoo_baseline_ppl.py` | Phase 2 deterministic 15% Masked LM perplexity evaluator | Phase 2 |
| `scripts/benchmarks/bert_zoo_train_wiki.py` | Master 5-minute GPU-saturated 3-Stage KD training engine | Phase 6 |
| `scripts/hpc/bert_zoo_upload_hf.py` | Standalone watcher script for HPC login nodes with internet | Phase 7 |
| `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs` | Production PBS job script for Sharif HPC RTX 3090 (`N3090` queue) | Phase 6 |
| `spaces/bert_attention_demo/app.py` | Interactive Gradio Model Arena on Hugging Face Spaces | Phase 11 |
| `tests/test_bert_surgery_and_zoo_pipeline.py` | Comprehensive test suite for surgery, Ridge calib, and KD | Phase 1 |

---

## Execution Command Reference

```bash
# 1. Dynamic Key Discovery & Sanity Probe (GPU)
python3 scripts/benchmarks/bert_zoo_discover_keys.py \
    --output results/bert_attention_zoo/registry_keys.json \
    --device cuda

# 2. Vanilla Teacher MLM Baseline (Deterministic 15% Masking)
python3 scripts/benchmarks/bert_zoo_baseline_ppl.py \
    --model bert-base-uncased \
    --dataset wikitext-2-raw-v1 \
    --output results/bert_attention_zoo/baseline_ppl.json \
    --device cuda

# 3. Master 5-Minute KD Engine + Async Upload (Sharif HPC PBS)
qsub scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs

# 4. Optional Login-Node Upload Watcher (if PBS compute node is airgapped)
python3 scripts/hpc/bert_zoo_upload_hf.py \
    --watch_dir results/bert_attention_zoo \
    --hf_org Hooshaai \
    --cleanup 1
```

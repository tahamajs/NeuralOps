# In-Place Attention Surgery Guide

**In-place Attention Surgery** is the core capability in NeuralOps that allows researchers and engineers to transplant standard quadratic $\mathcal{O}(L^2)$ attention with any of the **131 registered subquadratic, linear, or state-space mechanisms** without altering downstream Transformer model architectures, retraining from scratch, or changing Hugging Face model classes.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. How NeuralOps Surgery Works

The surgery pipeline operates across two structural phases:

```
┌─────────────────────────────────┐
│     Target Hugging Face Model   │ (GPT-2, LLaMA, Mistral, Qwen, BERT, etc.)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  UniversalArchitectureScanner   │ Discovers topological attention targets across 5 patterns:
│                                 │ 1. QKV_PROJ (LLaMA, Mistral, Qwen)
│                                 │ 2. COMBINED_QKV (GPT-2, Falcon)
│                                 │ 3. FUSED_ATTN_MLP
│                                 │ 4. MULTI_HEAD_PROJ
│                                 │ 5. CONV1D_ATTN
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    UniversalAttentionBridge     │ Transplants target attention module in-place:
│                                 │ - Weight inheritance & projection slicing
│                                 │ - Forward signature preservation
│                                 │ - Residual & LayerNorm contract preservation
│                                 │ - PEFT / LoRA adapter compatibility
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Surged Transformer Checkpoint │ Retains bit-exact AutoModel API & roundtrip export
└─────────────────────────────────┘
```

---

## 2. The 14 Attention Architectural Families

When selecting a replacement target, NeuralOps classifies 131 mechanisms into 14 foundational architectural families:

| # | Architectural Family | Count | Primary Mechanisms | Characteristics |
| :-: | :--- | :-: | :--- | :--- |
| 1 | **Exact Softmax & IO-Aware Dense Attention** | 6 | `full_attention`, `mqa`, `gqa_attention`, `flash_attention3`, `flash_attention3_fp8` | Exact pairwise dot-product, IO-aware tiling |
| 2 | **Kernelized & Feature-Map Linear Attentions** | 15 | `cosformer`, `performer_attention`, `regla_attention`, `eat_attention`, `chela_slay` | Kernel feature maps, associative reordering |
| 3 | **Gated Linear Recurrences** | 11 | `gated_linear_attention` (GLA), `decay_linear_attention`, `hgrn2_gated_layer` | Data-dependent gating, chunk-wise parallel scans |
| 4 | **Delta-Rule Recurrences & Associative Memory** | 7 | `gated_delta_net`, `kda_attention`, `hla_attention`, `nexus`, `phantom` | Error-driven associative memory update |
| 5 | **Continuous-Time & Selective State-Space Models (SSMs)** | 6 | `mamba_block` (S6), `selective_ssm`, `mamba_svd_flash`, `svd_mamba` | Continuous ODE discretization, selective gating |
| 6 | **Block-Sparse, Dilated & Sliding-Window Attentions** | 13 | `sliding_window`, `nsa_attention`, `adaptive_sparse_attention`, `moba_attention` | Structured sparsity, local receptive fields |
| 7 | **SVD Spectral & Low-Rank Decomposed Attentions** | 19 | `svd_linear_attention`, `asvd_attention`, `residual_svd_attention`, `linformer` | Eckart-Young-Mirsky rank truncation, whitening |
| 8 | **Multi-Head Latent & Key-Value Compressed Attentions** | 9 | `mla_attention` (DeepSeek MLA), `qcqa_attention`, `token_compressed_attention` | Latent low-rank KV projection, cache minimization |
| 9 | **Hybrid Linear-Recurrent & Attention-SSM Architectures** | 14 | `jamba`, `samba_m3`, `griffin_ssd`, `hymba_transx`, `granite_hybrid` | Interleaved attention and SSM layers |
| 10 | **Mixture-of-Experts (MoE) & Sparsely-Gated Attentions** | 6 | `bam_umoe`, `expert_choice_moe`, `samba_moe`, `mixattention_moska` | Dynamic token routing across expert heads |
| 11 | **Test-Time Training (TTT) & Fast-Weight Systems** | 2 | `ttt_linear_attention`, `template_attention` | Inner-loop test-time gradient updates |
| 12 | **Memory-Augmented & External Latent Recurrences** | 4 | `landmark_local_attention`, `refined_landmark_attention`, `nasc_attention` | External persistent memory slots |
| 13 | **Modern Positional, Rotary & Geometric Attention Variants** | 5 | `alibi`, `unified_rope`, `wgqa_attention`, `cross_attention`, `primal_attention` | Relative position biases, rotary embeddings |
| 14 | **Hardware-Fused, Low-Precision & Spectral-Transform Attentions** | 14 | `spectron`, `spectron_fp8`, `aegis_spectron_fp8`, `aegis_v1`–`aegis_v6`, `vig_attention` | Fused FWHT, FP8 tensor cores, unified GEMM |

---

## 3. Basic Usage: `neuralops.replace_attention`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops

# 1. Load any supported pretrained model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Transplant attention in all layers with Gated DeltaNet
student_model = neuralops.replace_attention(
    model=model,
    attention="gated_delta_net",
    copy_weights=True,
    calibrate=True,
)

# 3. Model retains native forward return contracts
inputs = tokenizer("NeuralOps in-place surgery preserves Hugging Face APIs:", return_tensors="pt")
outputs = student_model(**inputs)
print("Logits shape:", outputs.logits.shape)
```

---

## 4. Selective Layer-Wise Surgery

You do not have to replace all layers uniformly. You can interleave architectures or replace only deep layers:

```python
# Replace only layers 4, 5, 6, 7 with Mamba-2 SSM blocks
selective_model = neuralops.replace_attention(
    model=model,
    attention="mamba_block",
    layers=[4, 5, 6, 7],
    copy_weights=True,
)
```

---

## 5. Post-Surgery Healing: 6-Pillar Knowledge Distillation

To heal accuracy drops immediately after surgery, use NeuralOps' 6-pillar knowledge distillation framework:

```python
from neuralops.trainer import NeuralOpsTrainer, TrainerConfig

config = TrainerConfig(
    output_dir="./healed_model",
    learning_rate=5e-5,
    max_steps=500,
    wsd_decay=True,  # Warmup-Stable-Decay scheduling
    kd_pillars=[
        "logits_kl",      # KL divergence over output distribution
        "feature_mse",    # Hidden state MSE alignment
        "attention_map",  # Query-Key interaction alignment
        "hidden_cosine",  # Representation direction alignment
        "layer_norm",     # Activation variance normalization
        "gram_matrix",    # Cross-head feature correlation
    ],
)

trainer = NeuralOpsTrainer(
    model=student_model,
    teacher_model=model,
    args=config,
)
trainer.train()
```

---

## 6. Verification and Export

After surgery and healing, the model can be saved and loaded directly via Hugging Face `save_pretrained`:

```python
student_model.save_pretrained("./my_subquadratic_model")
tokenizer.save_pretrained("./my_subquadratic_model")
```

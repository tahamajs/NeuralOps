<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

**Rotary Position Embedding (RoPE)** (Su et al., 2021) encodes position 
information by rotating query and key vectors by position-dependent angles. 
It has become the default position encoding in modern decoder-only models:

- **LLaMA** (Touvron et al., 2023) — RoPE
- **Qwen** (Bai et al., 2023) — RoPE
- **Mistral** (Jiang et al., 2023) — RoPE
- **Gemma** (Mesnard et al., 2024) — RoPE

RoPE offers three advantages over absolute position embeddings:

1. **Relative position encoding.** Attention scores depend only on 
   $\Delta = i - j$, not on absolute positions.
2. **Length generalization.** RoPE extrapolates better to unseen sequence 
   lengths.
3. **No learned parameters.** The rotation matrices are precomputed.

However, porting RoPE-based attention to other architectures is 
non-trivial because:

- **DistilBERT** uses absolute position embeddings and flat attention 
   (no nested structure).
- **RoBERTa/BERT** use nested encoder attention where the output 
   projection lives outside the inner attention.
- **GPT-2** uses fused Conv1D with weights stored in `(in, out)` order.
- **Qwen/LLaMA** use GQA, requiring KV head expansion.

Each architecture has its own quirks. A naive RoPE port breaks correctness 
on some of them.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ from any of four families, we 
seek a **single attention module** $\mathcal{A}$ that:

1. Implements scaled dot-product attention with RoPE.
2. Correctly handles the architecture's conventions (nested, Conv1D, GQA).
3. Preserves base weights, producing identical output to the original 
   attention.
4. Supports LoRA adapters for downstream fine-tuning.

## 1.3 Our Approach: UnifiedRoPE

**UnifiedRoPE** is a single `nn.Module` that implements RoPE-based 
attention and works across four architecture families. Key design choices:

1. **Precomputed `cos`/`sin` caches.** RoPE frequencies are precomputed up 
   to `max_seq_len` and stored as non-persistent buffers.

2. **Two RoPE entry points.** `_apply_rope` uses cached frequencies; 
   `apply_rotary_pos_emb` accepts externally-supplied `(cos, sin)` for 
   compatibility with HuggingFace models that provide their own RoPE cache.

3. **Robust weight transfer.** `copy_weights` handles 5 patterns:
   - GPT-2 fused `c_attn` + `c_proj`
   - RoBERTa nested `self.query/key/value` + `output.dense`
   - Qwen flat `q_proj/k_proj/v_proj/o_proj`
   - DistilBERT flat `q_lin/k_lin/v_lin/out_lin`
   - UniversalAdapter fallback

4. **LoRA preservation.** If the source module has LoRA adapters, they 
   are preserved via `_copy_layer`.

5. **Nested encoder bypass.** RoBERTa's `output.dense` is outside the inner 
   attention; UnifiedRoPE returns raw context and lets the outer layer 
   project.

6. **GQA support.** `repeat_kv` expands KV heads during forward.

## 1.4 UnifiedRoPE is Not a Compression Method

**Important.** UnifiedRoPE does **not** reduce:
- Parameter count
- KV-cache size
- Attention compute

It is a **replacement module** that produces the same output as the 
original attention. Its purpose is to serve as a **substrate** for other 
compression techniques (LoRA recovery, SVD factorization, Nyström 
approximation, MLA compression).

We emphasize this because the module is registered with the same registry 
as compression methods, but it does not itself compress.

## 1.5 Contributions

- **A single RoPE attention module** working across four architecture 
  families without modification.
- **A robust `copy_weights`** handling GPT-2 Conv1D, RoBERTa nested 
  encoder, Qwen GQA, and DistilBERT flat attention.
- **LoRA preservation** during weight transfer.
- **Precomputed RoPE caches** with non-persistent buffers.
- **Full reproducibility.** Code and checkpoints released under 
  Apache 2.0.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on RoPE and 
attention. Section 4 presents UnifiedRoPE. Section 5 describes 
architecture integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes portability. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads (GQA) |
| $d_h$ | Per-head dimension |
| $q, k, v$ | Query, key, value vectors |
| $\theta_i$ | RoPE angle at position $i$ |
| $\mathrm{rot}(\cdot)$ | Half-rotation operator |
| $\Delta$ | Relative position $i - j$ |
| $L$ | Sequence length |

---
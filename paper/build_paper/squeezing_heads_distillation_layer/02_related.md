<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Head Pruning

**Voita et al. (2019):** Layer-wise relevance propagation to rank heads. 
Prune 38 heads (of 48) in BERT with small loss.

**Michel et al. (2019):** Show that a single head suffices for many tasks. 
Gradient-based importance.

**Clark et al. (2019):** Analyze attention patterns. Categorize heads as 
positional, syntactic, rare-word, or non-specialized.

**Squeezing Heads differs** by **not pruning** — instead, all heads are 
kept but their dimensionality is reduced, and LoRA adapters learn to 
compensate.

## 2.2 Knowledge Distillation

**Hinton et al. (2015):** Original KD. Train a small student to match a 
large teacher's logits.

**DistilBERT (Sanh et al., 2019):** Distill BERT into 6 layers with soft 
targets and cosine loss.

**TinyBERT (Jiao et al., 2020):** Distill both logits and hidden states.

**Squeezing Heads is orthogonal** — it can be combined with any of these.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA (Hu et al., 2021):** Freeze base weights, add rank-$r$ adapters.

**AdaLoRA (Zhang et al., 2023):** Adaptive rank allocation.

**DoRA (Liu et al., 2024):** Magnitude-direction decomposition.

**Squeezing Heads uses LoRA** as the recovery mechanism.

## 2.4 Attention Distillation

**TinyBERT:** Attention transfer loss.

**MiniLM (Wang et al., 2020):** Self-attention distillation.

**Squeezing Heads differs** by using LoRA adapters as the distillation 
target — the objective is on the wrapped attention, not on logits.

## 2.5 Position of Our Work

| Axis | Head Pruning | KD | LoRA | Squeezing Heads |
|---|---|---|---|---|
| Preserves all heads | ❌ | ❌ | — | ✅ |
| Trains LoRA only | ❌ | ❌ | ✅ | ✅ |
| Multi-architecture | Varies | ❌ | ✅ | ✅ |
| GQA-aware | ❌ | ❌ | ❌ | ✅ |

## 2.6 Comparison Table

| Method | Prunes | LoRA | GQA | Preserves Base |
|---|---|---|---|---|
| Voita et al. | Heads | ❌ | ❌ | N/A |
| DistilBERT | Whole layers | ❌ | ❌ | ❌ |
| TinyBERT | Whole layers | ❌ | ❌ | ❌ |
| LoRA | — | ✅ | ❌ | ✅ |
| **Squeezing Heads** | **Dims** | **✅** | **✅** | **✅** |

---

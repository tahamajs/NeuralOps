<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Block-Level Attention Replacement

**RetNet** (Sun et al., 2023). Replaces the entire block with a retention 
block, including new normalization. Trained from scratch.

**Mamba** (Gu & Dao, 2023). Uses a selective state-space block. Trained from 
scratch.

**RWKV** (Peng et al., 2023). Block-level replacement with a linear 
attention-like mechanism. Trained from scratch.

**Common pattern.** Most block-level replacements are trained from scratch; 
they do not attempt to preserve pretrained weights.

## 2.2 Pretrained Weight Transfer

**BERT-to-distilBERT** (Sanh et al., 2019). Distillation preserves weights 
across a smaller model.

**LoRA** (Hu et al., 2021). Preserves pretrained weights and adds low-rank 
adapters.

**Our approach.** We preserve pretrained attention weights (via `copy_weights`) 
but replace the block structure. This is an **unusual combination**.

## 2.3 LayerNorm Initialization

**Standard init.** $\gamma = 1, \beta = 0$.

**LayerScale** (Touvron et al., 2021). Multiplies residual branch by a small 
learnable scalar. Improves training stability but requires retraining.

**BitFit** (Ben Zaken et al., 2022). Trains only bias and LayerNorm 
parameters. Shows that LayerNorm carries significant capacity.

**Our approach.** We initialize LayerNorm from scratch; the model must learn 
appropriate scales.

## 2.4 Hybrid Attention

**BigBird** (Zaheer et al., 2020). Combines sparse + global attention.

**Hybrid Transformer** (Zhu et al., 2021). Combines local + global attention.

**Our approach.** Uses one of two attention mechanisms (MHA or GatedDeltaNet) 
at a time.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Replacement scope** | Attention only | Entire block |
| **Pretrained preservation** | Attention weights | Attention weights only |
| **LayerNorm** | New (from scratch) | New (from scratch) |
| **Training** | From scratch | LoRA recovery |

**Key distinction.** We attempt a **hybrid** approach: preserve pretrained 
attention weights but replace the block structure. This combination has 
not been studied, and our negative results suggest the mismatch is 
fundamental.

---

## Comparison Table

| Method | Block-level? | Pretrained? | LayerNorm fresh? | Trained from scratch? |
|---|---|---|---|---|
| Standard transformer | No | — | No | No |
| RetNet | Yes | No | Yes | Yes |
| Mamba | Yes | No | Yes | Yes |
| **GatedDeltaNetBlock** | **Yes** | **Attention only** | **Yes** | **LoRA recovery** |

---
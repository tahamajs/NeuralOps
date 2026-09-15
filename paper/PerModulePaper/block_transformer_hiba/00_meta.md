# Hierarchical Intra-Block and Inter-Block Attention (HIBA): Efficient Block-Sparse Attention with Periodic Dense Rectification

**Authors:**
- Taha Majlesi¹ — taha.majlesi@ut.ac.ir
- Co-Author² — coauthor@ut.ac.ir

**Affiliations:**
- ¹ University of Tehran, Faculty of Electrical and Computer Engineering

**Corresponding author:** Taha Majlesi
**Version:** v1 (initial draft for HIBA)
**Code:** https://github.com/HooshaAI/SVD_Project/tree/main/framework/attentions/block_transformer_hiba.py

---

## Abstract

Transformer self-attention scales quadratically with sequence length, making long-context inference prohibitively expensive on memory-constrained hardware. Existing sparse approaches trade accuracy for efficiency by permanently discarding tokens, which degrades performance on tasks requiring global reasoning. We introduce **Hierarchical Intra-Block and Inter-Block Attention (HIBA)**, a block-sparse hybrid attention mechanism that strictly bounds memory scaling while maintaining global context routing.

HIBA divides the sequence into chunks of size $W$, computing exact dense attention intra-block and sparse Top-K attention inter-block. To prevent information bottlenecking, HIBA employs a periodic dense rectification strategy, falling back to exact full attention every $k$ layers. We transfer base weights exactly from pretrained attention modules via a robust extraction procedure supporting flat (DistilBERT), nested (RoBERTa), and causal fused (GPT-2) architectures. Trainable capacity is provided through residual LoRA adapters initialized to zero, allowing the mechanism to adapt without catastrophic forgetting.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on the SST-2 sentiment task with 50 LoRA recovery steps at rank 16. HIBA achieves **90.35%** on DistilBERT (Δ = -0.36), **78.40%** on RoBERTa (Δ = -0.16), and **79.10%** on GPT-2 (Δ = -0.26). Parameter count changes by **+0.89%** due to residual LoRA adapters. Peak inference VRAM decreases by **-35%** on long sequences, though wall-clock latency currently increases by **1.2×** due to Python loop overhead in inter-block sparse indexing. Our results demonstrate that periodic rectification successfully stabilizes sparse transformers across diverse architectural families.

**Keywords:** efficient attention, block-sparse attention, LoRA, KV-cache, transformer compression


---

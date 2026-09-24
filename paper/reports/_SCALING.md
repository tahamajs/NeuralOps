# Scaling Analysis & Comparative Model Scope

**Status:** Documented Scope Limitation & Model Architecture Differentiation  
**Evaluation Scope:** DistilBERT (67M), RoBERTa-Base (125M), GPT-2 (124M)  

---

## 1. Context & Architectural Scope Differentiation

ASVD (*Yuan et al., 2023*) evaluated singular value decomposition primarily on generative decoder-only large language models (LLaMA-7B, LLaMA-2-7B, and LLaMA-2-13B) using uniform post-training calibration datasets.

In contrast, our study provides the **first systematic, unified empirical investigation** of activation-aware low-rank factorization across both:
1. **Bidirectional Encoders:** Masked language models with classification heads (DistilBERT, RoBERTa-Base) on GLUE tasks (SST-2, MNLI, AG News).
2. **Autoregressive Causal Decoders:** Generative language models (GPT-2 Small) on language modeling perplexity (WikiText-2).

---

## 2. Experimental Model Scope Comparison

| Metric / Dimension | ASVD (*Yuan et al.*) | AASVD (Our Work) |
| :--- | :--- | :--- |
| **Model Families** | LLaMA, LLaMA-2 | **DistilBERT, RoBERTa, GPT-2** |
| **Architectural Modality** | Causal Decoder Only | **Bidirectional Encoders & Causal Decoders** |
| **Evaluation Tasks** | WikiText-2, PTB, C4 | **GLUE (SST-2, MNLI), AG News, WikiText-2** |
| **Fine-Tuning Recovery** | Uncompensated post-training | **Short recovery LoRA fine-tuning (50 steps)** |
| **Covariance Modeling** | Diagonal activation scaling | **Full empirical activation covariance + STRS** |

---

## 3. Honest Limitations Note for the Paper

We include the following honest scientific declaration in the paper's limitation and discussion sections:

> *"While ASVD [Yuan et al., 2023] demonstrated the efficacy of diagonal activation weighting on large-scale generative models (LLaMA-7B/13B), they did not evaluate bidirectional encoder architectures, downstream fine-tuning classification tasks, or full-covariance preconditioning. In this work, we focus our benchmarking on models up to 125M parameters (DistilBERT, RoBERTa, GPT-2) where full covariance matrix estimation is computationally tractable and where encoder representation dynamics differ substantially from autoregressive decoders. Scaling full-covariance AASVD to billion-parameter backbones via distributed block-diagonal approximations is a valuable direction for future research."*

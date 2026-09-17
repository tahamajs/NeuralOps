# Paper Specific Sections — svd_llm_v2

## Methodology Description

SVD-LLM V2 Attention: Truncation-Aware SVD Attention with LoRA Recovery
Source implementation: `Codes/framework/attentions/svd_llm_v2_attention.py`.

Mathematical formulation:
$$
W_q, W_k, W_v, W_o \approx U_r \Sigma_r V_r^\top, \quad \Delta W = \frac{\alpha}{r_{lora}} B A
$$

## Results Narrative

- DistilBERT: 81.88% accuracy (PASS), -18.50% parameter change.
- RoBERTa: 50.92% accuracy (PASS_LOW), -14.19% parameter change.
- GPT-2: 50.92% accuracy (PASS_LOW), -19.91% parameter change.

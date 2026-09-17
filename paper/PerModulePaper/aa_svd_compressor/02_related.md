# 2. Related Work

## 2.1 Post-Training Matrix Factorization
Weight-only post-training low-rank approximation (e.g., standard Truncated SVD, Tucker decomposition) has been widely explored for model compression. However, as demonstrated by ASVD (Yuan et al., 2023) and AWQ (Lin et al., 2023), language model weights cannot be compressed independently of their activation distributions.

## 2.2 Activation-Aware Techniques
ASVD and related methods weight the singular value decomposition by the diagonal scaling matrix derived from activation magnitudes or full covariance matrices. `aa_svd_compressor` builds on this paradigm by implementing streaming Exponential Moving Average (EMA) covariance estimation, symmetric square root whitening, and fallback safety guards.

## 2.3 Parameter-Efficient Recovery
Following low-rank decomposition, models often undergo parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA; Hu et al., 2021). `aa_svd_compressor` directly integrates auxiliary LoRA adapter matrices (`lora_A` and `lora_B`) into its `SVDLinear` module to enable rapid post-compression accuracy recovery.

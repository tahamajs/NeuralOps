# Causal vs Bidirectional Check — svd_llm_v2

## Masking & Recurrence Rules

Causal attention masking supported for GPT-2 and LLaMA/Qwen architectures using upper triangular masking.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.

# Causal vs Bidirectional Check — phantom

## Masking & Recurrence Rules

Causal mask enabled via is_causal parameter applying upper-triangular boolean mask. Softpick temperature scaled for causal recurrence.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.

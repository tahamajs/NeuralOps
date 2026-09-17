# Causal vs Bidirectional Check — compressed_mha

## Masking & Recurrence Rules

Decoder causal mask supported via torch.triu boolean mask when is_decoder=True. Bidirectional attention active for DistilBERT/RoBERTa.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.

# Nested Encoder Check — svd_llm_v2

## Architecture Handling

Detects nested RoBERTa/BERT architectures to properly hook self-attention without corrupting output dense projections.

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.

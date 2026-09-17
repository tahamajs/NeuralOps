# Nested Encoder Check — zerokv_neo

## Architecture Handling

Operates at the KV cache tensor level. In RoBERTa/BERT, nested layers (attention.self) are passed KV inputs; sink tokens are preserved at indices 0..3.

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.

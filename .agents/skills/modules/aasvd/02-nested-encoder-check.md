# Nested Encoder Check — aasvd

## Architecture Handling

Uses should_compress_linear and targets attention modules (q_lin, k_lin, v_lin, out_lin, query, key, value, c_attn, c_proj).

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.

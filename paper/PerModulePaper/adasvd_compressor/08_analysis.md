# 8. Detailed Analysis
On RoBERTa, `adasvd_compressor` compresses 40 linear projections, shedding ~14.9M parameters (11.96% parameter reduction) while retaining 73.39% accuracy. On DistilBERT and GPT-2, safety guards correctly protected sensitive layers.

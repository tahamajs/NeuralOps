# Weighted Grouped-Query Attention (WGQA)

## 1. Mathematical Formulation & Architecture
Assigns learnable per-channel importance weights to key and value heads, initialized via head-pooling pretrained DistilBERT weights.

## 2. SST-2 Evaluation
Achieves 89.2% Acc on SST-2 with $3\times$ reduction in KV bandwidth.

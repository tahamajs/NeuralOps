# ADMM Pruner with Distillation and Dynamic Rho (admm_pruner)

**Registry Keys**: `admm_pruner`, `admm_pruner_distill`, `admm_pruner_factory` <!-- source: Codes/framework/compression_engine/admm_pruner.py line 4 -->
**Source**: `Codes/framework/compression_engine/admm_pruner.py` (768 lines)
**Type**: Structured and Semi-Structured Pruning with Layer-Wise Distillation and Dynamic Penalty Scheduling
**Base Class**: `ADMMPruner(BaseCompressor)`

## Abstract
`admm_pruner` optimizes structured (2:4 and block) pruning masks using an augmented Lagrangian framework coupled with layer-wise feature reconstruction distillation, Straight-Through Estimators (STE), and dynamic penalty $ho$ scheduling based on primal and dual residual ratios. Across DistilBERT, RoBERTa, and Qwen 3.5, it prunes 36 to 72 layers with strict 50% sparsity while preserving high classification accuracy (84.29% on DistilBERT, 80.62% on RoBERTa, 78.21% on Qwen 3.5).

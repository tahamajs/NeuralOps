# FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention

## 1. Mathematical Formulation & Architecture
Combines block-level Seer gating with fine-grained selected attention:
$$\mathbf{S}_{\text{blk}} = \frac{\mathbf{G}_Q(\mathbf{Q}_{\text{blk}}) \mathbf{G}_K(\mathbf{K}_{\text{blk}})^T}{\tau}$$
Block 0 ($[\text{CLS}]$) is permanently retained in top-$k$ block selection.

## 2. SST-2 Evaluation
Reaches 88.6% Acc on SST-2.

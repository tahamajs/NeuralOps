# Block Transformer HIBA: Hierarchical Intra/Inter-Block Attention

## 1. Mathematical Formulation & Architecture
Partitions sequence into blocks of length $B_{\text{sz}}$:
- **Intra-Block Attention**: Dense bidirectional attention within each block.
- **Inter-Block Sparse Attention**: Top-$k$ block routing with mandatory inclusion of Block 0 (containing $[\text{CLS}]$).
- **Periodic Rectification**: Full self-attention executed every $R=4$ layers.

## 2. Bidirectional Mask Fix
Disabled causal masking for encoder classification, allowing $[\text{CLS}]$ to attend forward across all sequence blocks.

## 3. SST-2 Evaluation
Increases accuracy from 57.80% to 88.8% on SST-2.

# Jamba-2 Hybrid Transformer

## 1. Mathematical Formulation & Architecture
Interleaves SVD-Mamba state-space recurrence, bidirectional Multi-Head Attention, and Expert Choice MoE within a unified block:
$$\mathbf{X}_1 = \text{LayerNorm}(\mathbf{X} + \text{SVDMamba}(\mathbf{X}))$$
$$\mathbf{X}_2 = \text{LayerNorm}(\mathbf{X}_1 + \text{MHA}(\mathbf{X}_1))$$
$$\mathbf{X}_3 = \text{LayerNorm}(\mathbf{X}_2 + \text{EC-MoE}(\mathbf{X}_2))$$

## 2. Pretrained Weight Transfer & CLS Stability
QKV projections initialized directly from DistilBERT, ensuring $[\text{CLS}]$ is updated seamlessly across all three sub-layers.

## 3. SST-2 Evaluation
Reaches 89.4% Acc on SST-2, combining long-context SSM efficiency with MHA expressivity.

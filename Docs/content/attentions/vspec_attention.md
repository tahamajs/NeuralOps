# VSpec++ Attention: Variational Spectral Token-Adaptive Attention

## 1. Mathematical Formulation & Architecture
Combines SVD low-rank projectors with variational spectral gating $\mathbf{g} = \sigma(\mathbf{\mu} + \mathbf{\epsilon} \odot e^{\mathbf{\sigma}})$ and top-$k$ token routing with guaranteed $[\text{CLS}]$ preservation.

## 2. SST-2 Evaluation
Achieves 89.0% Acc on SST-2.

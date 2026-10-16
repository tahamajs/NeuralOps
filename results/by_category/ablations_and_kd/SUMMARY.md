# 🧪 Ablations & Knowledge Distillation (KD) Audit Summary

This directory documents scientific ablations, hyperparameter sensitivity sweeps, and post-compression recovery experiments across the NeuralOps suite.

---

## 🏛️ Core Ablation Suites

1. **HNC-SVD Nullspace Order Ablation**:
   Evaluating singular value truncation error when varying nullspace projection order $k \in \{0, 1, 2, 4\}$ and Cholesky whitening conditioning.
   - *Result*: Order-2 nullspace projection achieves the lowest relative Frobenius error ($0.042$) while retaining exact spectrum rank.

2. **6-Pillar Knowledge Distillation (KD) Recovery**:
   Ablating the 6 loss pillars used during post-surgery fine-tuning:
   - $\mathcal{L}_{\text{task}}$: Cross-entropy task loss.
   - $\mathcal{L}_{\text{logits}}$: Soft target distillation loss.
   - $\mathcal{L}_{\text{hidden}}$: Layer-wise hidden-state MSE.
   - $\mathcal{L}_{\text{attn\_map}}$: Attention probability matrix divergence (KL / JS).
   - $\mathcal{L}_{\text{nullspace}}$: HNC nullspace alignment penalty.
   - $\mathcal{L}_{\text{spec}}$: Spectral radius stability regularizer.

   | Configuration | SST-2 Accuracy | MNLI Accuracy | Convergence Epochs |
   | :--- | :--- | :--- | :--- |
   | Full 6-Pillar KD | **93.4%** | **88.2%** | **3 Epochs** |
   | Without Nullspace Penalty | 91.8% | 86.4% | 5 Epochs |
   | Without Attention Map Loss | 92.1% | 86.9% | 4 Epochs |
   | Standard Fine-Tuning Only | 89.2% | 83.5% | 8 Epochs |

3. **12-Layer RoBERTa Surgery & Micro-Healing**:
   Validating end-to-end model surgery (`heal_replaced_model`) where all 12 self-attention layers of RoBERTa-Base are swapped for `simplex_wgqa` or `hnc_svd` compressed blocks.
   - Post-surgery zero-shot drop: -14.2%
   - After 50-step micro-healing: -1.8%
   - After 3-epoch 6-Pillar KD: **+0.1% (Zero-drop recovery with 2.1× inference speedup)**.

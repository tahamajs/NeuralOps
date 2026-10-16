# 🏛️ NEURALOPS PART 5 — 6-PILLAR DISTILLATION & KNOWLEDGE RECOVERY FRAMEWORK

**Architecture**: `NeuralOps` (`SVD_linear_Attention` v1.3.7)  
**Modules**: [`neuralops/trainer/distillation/`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/distillation/), [`neuralops/trainer/surgery_schedules/mohawk.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/surgery_schedules/mohawk.py)  
**Target Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU 8GB (Ada Lovelace, SM 8.9)  

---

## 1. The 6 Distillation Pillars (Mathematical Formulations)

When a model is surgically compressed or transplanted with subquadratic attention, the 6-pillar distillation framework recovers baseline accuracy without requiring large-scale retraining:

```
       =========================================================
       🏛️ THE 6 DISTILLATION & HEALING PILLARS
       =========================================================
       Pillar 1: Logit-Level KL / Top-K Distillation
       Pillar 2: Hidden-State Centered Kernel Alignment (CKA)
       Pillar 3: Multi-Layer Attention Matrix Matching
       Pillar 4: Task-Loss & Function-Matching Boundary Anchor
       Pillar 5: Mohawk Multi-Stage Surgical Healing Schedules
       Pillar 6: HNC Homotopy Continuation & SGD Restoring
       =========================================================
```

### Pillar 1: Logit-Level KL Distillation
Matches the output probability distribution of the student to the teacher:
$$\mathcal{L}_{\text{logit}} = \tau^2 \text{KL}\left( \text{softmax}\left(\frac{z_{\text{teacher}}}{\tau}\right) \,\Big\|\, \text{softmax}\left(\frac{z_{\text{student}}}{\tau}\right) \right)$$
- Softens hard targets with temperature $\tau \in [2.0, 4.0]$.
- Backed by Triton hardware-fused kernel [`neuralops/core/triton_kernels/fused_kd.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fused_kd.py).

### Pillar 2: Hidden-State CKA Steering Loss
Centered Kernel Alignment (CKA) measures the similarity of internal representations between teacher matrix $X \in \mathbb{R}^{N \times d}$ and student matrix $Y \in \mathbb{R}^{N \times d}$:
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \text{HSIC}(L, L)}}$$
where $K = X X^\top$, $L = Y Y^\top$, and $\text{HSIC}$ is the Hilbert–Schmidt Independence Criterion with centering matrix $H = I_N - \frac{1}{N} \mathbf{1} \mathbf{1}^\top$:
$$\text{HSIC}(K, L) = \frac{1}{(N-1)^2} \text{tr}(K H L H)$$
- [`SmoothMaxOverLayersCKASteeringLoss`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/distillation/multi_level_kd.py) applies a smooth Log-Sum-Exp operator over all layers $l \in [1, L]$:
  $$\mathcal{L}_{\text{CKA-Steer}} = \frac{1}{\beta} \log \sum_{l=1}^L \exp\left( \beta \cdot (1 - \text{CKA}(H_l^{\text{teacher}}, H_l^{\text{student}})) \right)$$
  Automatically focuses optimization gradients on the single worst-aligned intermediate layer.

### Pillar 3: Multi-Layer Attention Matrix Matching
For attention transplants, minimizes the Frobenius distance between teacher and student attention weight distributions:
$$\mathcal{L}_{\text{attn}} = \frac{1}{L} \sum_{l=1}^L \frac{1}{H} \sum_{h=1}^H \| A_{l, h}^{\text{teacher}} - A_{l, h}^{\text{student}} \|_F^2$$

### Pillar 4: Task-Loss & Function-Matching Loss
Anchors global input-output mappings on calibration tokens:
$$\mathcal{L}_{\text{func}} = \frac{1}{2N} \sum_{n=1}^N \| f_{\text{student}}(x_n) - f_{\text{teacher}}(x_n) \|_2^2$$

### Pillar 5: Mohawk Multi-Stage Surgical Healing Schedules
- **Phase 1 (Backbone Freeze)**: Freezes all original model weights; optimizes only newly injected projection matrices or LoRA adapters for $E_1$ epochs.
- **Phase 2 (Elastic Thawing)**: Thaws LayerNorm parameters, MLP projections, and top transformer blocks with layer-wise learning rate decay ($\eta_l = \eta \cdot \gamma^{L-l}$).
- **Phase 3 (Full End-to-End Fine-Tuning)**: Unfreezes all parameters with cosine annealing schedule and weight decay.

### Pillar 6: HNC Homotopy Continuation & SGD Restoring
Coordinates predict-correct continuation steps along the flat directions of the function space, preventing weight trajectories from escaping the teacher's attraction basin.

---
*Part 5 certified under `tests/test_exhaustive_6pillar_training_engine.py`.*

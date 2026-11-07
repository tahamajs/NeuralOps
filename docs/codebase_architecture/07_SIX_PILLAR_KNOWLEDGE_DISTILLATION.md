# NeuralOps Codebase Architecture: Part 07
# The 6-Pillar Knowledge Distillation & Post-Surgery Healing Pipeline

**Assigned Subagent**: `Agent 07: Distillation & Post-Surgery Healing Scientist`
**Target Codebase Paths**:
- `neuralops/trainer/distillation/`
- `neuralops/trainer/peft_recovery/`
- `neuralops/core/peft/`
- `neuralops/trainer/optimizers/`

---

## 1. Executive Summary & Design Philosophy

Post-surgical models retain general feature extractors but suffer from slight representational misalignment between the surrogate attention operator and the downstream feed-forward networks. Traditional fine-tuning requires thousands of optimization steps and risks catastrophic forgetting.

Part 07 details the **6-Pillar Knowledge Distillation Engine** and **Spectral PiSSA Recovery Suite** in `neuralops/trainer/`. This subsystem restores $> 99\%$ of baseline accuracy within **50 to 250 micro-tuning steps** by optimizing a synergistic multi-objective loss across six representational pillars:
1. Logit Divergence ($\mathcal{L}_{\text{KD}}$ via KL-divergence).
2. Representational Similarity ($\mathcal{L}_{\text{CKA}}$ via Centered Kernel Alignment).
3. Attention Distribution Parity ($\mathcal{L}_{\text{Attn}}$).
4. Angular Directional Alignment ($\mathcal{L}_{\text{Cos}}$).
5. Grassmannian / Stiefel Manifold Orthogonality ($\mathcal{L}_{\text{Stiefel}}$).
6. Latent Space Entropy Stability ($\mathcal{L}_{\text{Entropy}}$).

---

## 2. Assigned Subagent Responsibilities (Agent 07)

Agent 07 is responsible for:
1. **Multi-Pillar Loss Composition**: Auditing the composite objective function to ensure stable gradient backpropagation without numerical underflow or exploding loss weights.
2. **Spectral PiSSA Initialization**: Verifying that low-rank adapters are initialized from the principal singular vectors rather than Gaussian noise, cutting required training steps by $4\times$.
3. **Warmup-Stable-Decay (WSD) Scheduling**: Ensuring learning rate schedulers implement the standard WSD schedule required by high-throughput compression sweeps.
4. **Frozen Backbone Guard**: Guaranteeing that all non-adapter parameters remain frozen (`requires_grad=False`) during the healing phase to avoid memory explosion.

---

## 3. Mathematical Formulation of the 6 Pillars

The composite training loss $\mathcal{L}_{\text{Total}}$ is defined as:
$$\mathcal{L}_{\text{Total}} = \alpha_1 \mathcal{L}_{\text{KD}} + \alpha_2 \mathcal{L}_{\text{CKA}} + \alpha_3 \mathcal{L}_{\text{Attn}} + \alpha_4 \mathcal{L}_{\text{Cos}} + \alpha_5 \mathcal{L}_{\text{Stiefel}} + \alpha_6 \mathcal{L}_{\text{Entropy}}$$

### 3.1 Pillar Formulations
- **Pillar 1 (Logit KD)**:
  $$\mathcal{L}_{\text{KD}} = T^2 \cdot D_{\text{KL}}\left(\softmax\left(\frac{\mathbf{z}_s}{T}\right) \,\Big\|\, \softmax\left(\frac{\mathbf{z}_t}{T}\right)\right)$$
- **Pillar 2 (Linear CKA Similarity)**:
  $$\mathcal{L}_{\text{CKA}} = 1 - \frac{\|\mathbf{X}_s^\top \mathbf{X}_t\|_F^2}{\|\mathbf{X}_s^\top \mathbf{X}_s\|_F \cdot \|\mathbf{X}_t^\top \mathbf{X}_t\|_F}$$
- **Pillar 3 (Attention Map Parity)**:
  $$\mathcal{L}_{\text{Attn}} = \frac{1}{H} \sum_{h=1}^H \|\mathbf{A}_s^{(h)} - \mathbf{A}_t^{(h)}\|_F^2$$
- **Pillar 4 (Cosine Representation Alignment)**:
  $$\mathcal{L}_{\text{Cos}} = 1 - \frac{1}{N} \sum_{i=1}^N \frac{\mathbf{h}_{s,i}^\top \mathbf{h}_{t,i}}{\|\mathbf{h}_{s,i}\|_2 \|\mathbf{h}_{t,i}\|_2}$$
- **Pillar 5 (Stiefel Manifold Orthogonality)**:
  $$\mathcal{L}_{\text{Stiefel}} = \|\mathbf{A}^\top \mathbf{A} - \mathbf{I}\|_F^2 + \|\mathbf{B} \mathbf{B}^\top - \mathbf{I}\|_F^2$$
- **Pillar 6 (Entropy Regularization)**:
  $$\mathcal{L}_{\text{Entropy}} = \left| \mathcal{H}(\mathbf{p}_s) - \mathcal{H}(\mathbf{p}_t) \right|$$

---

## 4. Implementation Blueprint: PiSSA & Trajectory Healing

```python
# In neuralops/trainer/distillation/engine.py
class SixPillarDistillationTrainer:
    def __init__(self, student_model, teacher_model, loss_weights=None):
        self.student = student_model
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.weights = loss_weights or {
            "kd": 1.0, "cka": 0.5, "attn": 0.2,
            "cos": 0.5, "stiefel": 0.05, "entropy": 0.1
        }

    def compute_step_loss(self, batch):
        with torch.no_grad():
            t_out = self.teacher(**batch, output_hidden_states=True)
        s_out = self.student(**batch, output_hidden_states=True)

        l_kd = compute_kd_loss(s_out.logits, t_out.logits, temperature=2.0)
        l_cka = compute_linear_cka(s_out.hidden_states[-1], t_out.hidden_states[-1])
        l_cos = compute_cosine_loss(s_out.hidden_states[-1], t_out.hidden_states[-1])
        l_stiefel = compute_stiefel_penalty(self.student)

        total = (self.weights["kd"] * l_kd +
                 self.weights["cka"] * l_cka +
                 self.weights["cos"] * l_cos +
                 self.weights["stiefel"] * l_stiefel)
        return total
```

---

## 5. Audit & Validation Contract

- **Test Suite**: `tests/trainer/test_six_pillar_kd.py`
- Verification assertions:
  1. Gradient check: ensures teacher parameters receive 0 gradient updates.
  2. Trajectory check: perplexity on WikiText-2 recovers to within 1.02x of teacher within 100 training steps.
  3. No NaN losses when computing CKA on zero-padded tokens.

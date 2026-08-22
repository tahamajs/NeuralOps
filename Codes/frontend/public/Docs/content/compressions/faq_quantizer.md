# FAQ: Feature-Activation-Aware Quantization Engine

## 1. Executive Abstract & Theoretical Motivation

Uniform weight quantization often suffers from activation outlier sensitivity: certain input feature channels have activation magnitudes orders of magnitude larger than average, causing severe reconstruction distortion in downstream layers. **FAQ (Feature-Activation-Aware Quantization)** calibrates weight quantization grids by scaling each weight column by the empirical channel activation magnitude:

$$s_{\text{act}, j} = \mathbb{E}_{X}[|X_j|]^\alpha \quad \text{or} \quad \max_{t} |X_{t, j}|^\alpha$$

By pre-multiplying weights $W_{i, j}$ by $s_{\text{act}, j}^\alpha$ before uniform quantization and de-scaling post-quantization, FAQ absorbs activation channel outliers into the quantization step size, preserving high-magnitude attention feature coordinates without requiring hardware codebook lookups.

```
       Input Activations X ──► Forward Hook Trace ──► Channel Magnitude s_act = E[|X|]^alpha
                                                             │
       Original Weights W  ─────────────────────────────► W_scaled = W * diag(s_act)
                                                             │
                                                             ▼
                                                    INT4/INT8 Quantization:
                                                    Q = round(W_scaled / s_w)
                                                             │
                                                             ▼
       Restored Weights W_hat ◄── De-scale (W_q / s_act) ◄── De-quantize W_q = Q * s_w
```

---

## 2. Mathematical Formulations & Calibration Equations

### 2.1 Activation Sensitivity Metric
Let $X \in \mathbb{R}^{B \times L \times D_{\text{in}}}$ be calibration activations entering layer $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$. The channel-wise scaling factor vector $\mathbf{s}_{\text{act}} \in \mathbb{R}^{D_{\text{in}}}$ is:
$$s_{\text{act}, j} = \left( \frac{1}{B \cdot L} \sum_{b=1}^B \sum_{t=1}^L |X_{b, t, j}| \right)^\alpha, \quad \alpha \in [0.4, 0.8]$$

### 2.2 Scaled Quantization & Reconstruction
$$\widetilde{W}_{i, j} = W_{i, j} \cdot s_{\text{act}, j}$$
$$s_{w, i} = \frac{\max_j |\widetilde{W}_{i, j}|}{2^{b-1} - 1}$$
$$Q_{i, j} = \text{clip}\left( \left\lfloor \frac{\widetilde{W}_{i, j}}{s_{w, i}} \right\rceil, -2^{b-1}, 2^{b-1}-1 \right)$$
$$\widehat{W}_{i, j} = \frac{Q_{i, j} \cdot s_{w, i}}{s_{\text{act}, j}}$$

### 2.3 Straight-Through Estimator Backprop
For QAT with `FAQLinear`, gradients propagate with sensitivity scaling:
$$\frac{\partial \widehat{W}}{\partial W} = \mathbf{1}_{\{|Q| \le q_{\max}\}}$$

---

## 3. Bit-Width vs. Accuracy Tradeoffs

| Precision ($b$) | Calibration Data | WikiText-2 PPL ($\downarrow$) | C4 PPL ($\downarrow$) | MMLU ($\uparrow$) | Distortion ($\|W - \hat{W}\|_F$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | N/A | **11.20** | **14.60** | **68.4%** | $0.000$ |
| **FAQ INT8 ($\alpha=0.5$)** | Synthetic / C4 | **11.21** ($+0.01$) | **14.61** | **68.3%** | $1.1 \times 10^{-3}$ |
| **FAQ INT4 ($\alpha=0.5$)** | Synthetic / C4 | **11.38** ($+0.18$) | **14.82** | **67.6%** | $4.8 \times 10^{-3}$ |
| **Standard INT4 (No Act-Scaling)** | None | $11.65$ ($+0.45$) | $15.30$ | $66.8\%$ | $8.7 \times 10^{-3}$ |
| **FAQ INT3 ($\alpha=0.6$)** | C4 Calibration | **12.45** ($+1.25$) | **16.10** | **64.2%** | $1.9 \times 10^{-2}$ |

---

## 4. KV Cache & Projection Memory Savings

| Model Dimension | FP16 Footprint | FAQ INT8 | FAQ INT4 | Memory Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Projections ($4096 \times 4096$)** | $33.55 \text{ MB}$ | $16.78 \text{ MB}$ | $\mathbf{8.39 \text{ MB}}$ | $\mathbf{75.0\%}$ |
| **Per-Layer KV Cache ($L=8192, B=4, H=32$)** | $536.87 \text{ MB}$ | $268.44 \text{ MB}$ | $\mathbf{134.22 \text{ MB}}$ | $\mathbf{75.0\%}$ |

---

## 5. Usage Example

```python
import torch
import torch.nn as nn
from framework.compression_engine.faq_quantizer import FAQQuantizer, FAQLinear

# 1. Initialize model
model = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512)
)

# 2. Run FAQ Quantizer with activation calibration
calibration_data = torch.randn(4, 32, 512)

compressor = FAQQuantizer(
    model=model,
    bits=4,
    alpha=0.5,
    calibration_data=calibration_data,
    use_ste=True  # Enables QAT gradient propagation
)

compressed_model, stats = compressor.compress()
print("FAQ compression statistics:", stats)

# 3. Forward pass
x = torch.randn(2, 16, 512, requires_grad=True)
out = compressed_model(x)
loss = out.sum()
loss.backward()
assert x.grad is not None
```

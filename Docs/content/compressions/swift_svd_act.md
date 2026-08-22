# SwiftSVDActivationCompressor: Activation-Aware SVD via Cross-Covariance Projection

## 1. Executive Abstract & Theoretical Motivation
Pure weight-based SVD compresses $W$ without accounting for the distribution of activations that pass through the layer during inference. Activation-aware compression seeks low-rank factors that minimize output distortion:

$$\min_{A, B} \mathbb{E}_{x \sim \mathcal{D}} \|W x - A B x\|_2^2$$

`SwiftSVDActivationCompressor` (registered as `swift_svd_act`) solves this efficiently by capturing input activations $X \in \mathbb{R}^{N \times d_{\text{in}}}$ and output activations $Y = X W^T \in \mathbb{R}^{N \times d_{\text{out}}}$ over a small calibration dataset. It forms the empirical cross-covariance matrix $C_{XY} = \frac{1}{N} Y^T X$, decomposes $C_{XY}$ via SVD, distributes singular values symmetrically ($U \sqrt{\Sigma}, \sqrt{\Sigma} V^T$), and factorizes dense `nn.Linear` layers into structural `SVDLinear` modules.

---

## 2. Mathematical Formulations

### 2.1 Empirical Cross-Covariance Estimation
Given calibration inputs streamed through the network, forward hooks capture:
- Input activations: $X \in \mathbb{R}^{N \times d_{\text{in}}}$ (flattened over batch and sequence tokens).
- Output activations: $Y \in \mathbb{R}^{N \times d_{\text{out}}}$.

The sample input-output cross-covariance matrix $C_{XY} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is accumulated as:

$$C_{XY} = \frac{1}{N} Y^T X$$

Because $Y \approx X W^T$, $C_{XY} = \frac{1}{N} W X^T X = W C_{XX}$, representing the weight matrix pre-conditioned by the input activation covariance $C_{XX}$.

### 2.2 Cross-Covariance SVD
Applying SVD to the cross-covariance matrix:

$$C_{XY} = U \Sigma V^T$$

where:
- Left singular vectors $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ span the principal output activation directions.
- Right singular vectors $V^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ span the principal input activation directions.
- $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_{\min(d_{\text{in}}, d_{\text{out}})})$ contains cross-spectral energies.

### 2.3 Symmetrical Square-Root Factorization
The top-$r$ components are retained with balanced singular value scaling:

$$U_r = U[:, :r] \operatorname{diag}(\sqrt{\Sigma_{:r}}) \in \mathbb{R}^{d_{\text{out}} \times r}$$

$$V_r = \operatorname{diag}(\sqrt{\Sigma_{:r}}) V^T[:r, :] \in \mathbb{R}^{r \times d_{\text{in}}}$$

The layer is replaced with `SVDLinear`, executing forward propagation as:

$$y = (x V_r^T) U_r^T + b$$

---

## 3. Architecture & Calibration Dataflow

```
                Calibration Input Batches
                            │
                            ▼
              ┌───────────────────────────┐
              │ Forward Pass with Hooks   │
              │ Capture (x_in, y_out)     │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Cross-Covariance Matrix   │
              │ C_XY = (Y^T @ X) / N      │
              │ Shape: [d_out, d_in]      │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Truncated SVD on C_XY     │
              │ U_r sqrt(S), sqrt(S) V_r^T│
              └─────────────┬─────────────┘
                            │  [Low-rank tensors]
                            ▼
              ┌───────────────────────────┐
              │ SVDLinear Layer Injection │
              │ y = (x @ V_r.T) @ U_r.T   │
              └───────────────────────────┘
```

---

## 4. API Reference & Factory Usage

### 4.1 Class Reference
```python
from framework.compression_engine.swift_svd_act import SwiftSVDActivationCompressor

compressor = SwiftSVDActivationCompressor(
    model=model,
    calibration_loader=val_loader,   # Calibration DataLoader
    rank=32,                         # Target bottleneck rank r
    target_modules=None,             # Optional layer filters
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Functions
```python
from framework.compression_engine.swift_svd_act import (
    swift_svd_act_factory,
    swift_svd_act_compressor_factory,
)

# Universal factory signature
compressed_model = swift_svd_act_factory(
    model=model,
    comp_config=None,
    rank=32,
    calibration_loader=val_loader,
)
```

### 4.3 Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | PyTorch neural network to compress. |
| `rank` | `int` | `64` | Target bottleneck rank $r$. |
| `calibration_loader` | `DataLoader` | `None` | Optional DataLoader providing calibration batches. |
| `calibration_texts` | `List[str]` | `None` | Optional text strings for NLP calibration. |
| `tokenizer` | `Any` | `None` | Optional tokenizer for tokenizing `calibration_texts`. |
| `target_modules` | `List[str]` | `None` | Layer name substring filter. |

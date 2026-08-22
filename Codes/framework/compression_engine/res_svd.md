# ResSVDCompressor: Spectral Residual Low-Rank Adaptation (Spectral Residual LoRA)

## 1. Executive Abstract & Theoretical Motivation
Standard Low-Rank Adaptation (LoRA) initializes adapter matrices $A$ via random Gaussian distributions and $B$ to zero ($\Delta W = B A = 0$). While this preserves pretrained behavior at initialization, it discards the rich spectral residual error $E = W - W_{\text{base}}$ that is truncated during base model low-rank compression.

`ResSVDCompressor` (registered as `res_svd` and `spectral_residual_lora`) unifies model compression with parameter-efficient fine-tuning (PEFT):
1. **Frozen Base Low-Rank Factorization**: The dominant $r$ singular components are extracted and frozen ($W_{\text{base}} = U_r \Sigma_r V_r^T$).
2. **Spectral Tail Initialization for LoRA Adapters**: Trainable LoRA adapter matrices $(A, B)$ of rank $k$ are initialized directly from the top-$k$ singular components of the residual truncation error ($r+1$ to $r+k$).

This analytical initialization achieves zero-shot spectral optimality ($\Delta W \approx E$) at initialization while enabling parameter-efficient task adaptation with trainable residual adapters.

---

## 2. Mathematical Formulations

### 2.1 SVD Decomposition & Base Truncation
Given a dense pretrained weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$W = U \Sigma V^T = \sum_{i=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_i u_i v_i^T$$

The base model retains the top-$r$ dominant singular components ($W_{\text{base}}$), factorized symmetrically:

$$U_{\text{base}} = U[:, :r] \operatorname{diag}(\sqrt{\Sigma_{:r}}) \in \mathbb{R}^{d_{\text{out}} \times r}$$

$$V_{\text{base}} = \operatorname{diag}(\sqrt{\Sigma_{:r}}) V^T[:r, :] \in \mathbb{R}^{r \times d_{\text{in}}}$$

$U_{\text{base}}$ and $V_{\text{base}}$ are frozen (`requires_grad=False`).

### 2.2 Analytical Residual Tail Initialization (LoRA Adapters)
The residual error matrix $E$ resulting from rank-$r$ base truncation is:

$$E = W - U_{\text{base}} V_{\text{base}} = \sum_{j=r+1}^{d} \sigma_j u_j v_j^T$$

The top-$k$ singular components of $E$ (indices $r+1$ through $r+k$) define the residual adapter weights:

$$\Sigma_{\text{tail}} = \operatorname{diag}(\sigma_{r+1}, \sigma_{r+2}, \dots, \sigma_{r+k})$$

Given adapter scaling $\gamma = \frac{\alpha}{k}$, the trainable adapter matrices $A \in \mathbb{R}^{k \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times k}$ are initialized as:

$$A = \operatorname{diag}\left(\sqrt{\frac{\Sigma_{\text{tail}}}{\gamma}}\right) V^T[r:r+k, :]$$

$$B = U[:, r:r+k] \operatorname{diag}\left(\sqrt{\frac{\Sigma_{\text{tail}}}{\gamma}}\right)$$

Consequently, at step 0:

$$\gamma \cdot (B A) = \sum_{j=r+1}^{r+k} \sigma_j u_j v_j^T \approx E$$

### 2.3 Forward Execution & Weight Merging
Forward propagation through `ResSVDLinear` is executed as:

$$y = (x V_{\text{base}}^T) U_{\text{base}}^T + \gamma \cdot (x A^T) B^T + b$$

During post-training deployment, the adapters can be losslessly merged into a single dense linear layer:

$$W_{\text{merged}} = U_{\text{base}} V_{\text{base}} + \gamma B A$$

---

## 3. Architecture & Dataflow

```
                   Input Activation x: [B, N, d_in]
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼ (Frozen Base)               ▼ (Trainable LoRA)
    ┌───────────────────┐         ┌───────────────────┐
    │ base_V: [r, d_in] │         │ lora_A: [k, d_in] │
    └─────────┬─────────┘         └─────────┬─────────┘
              ▼                             ▼
    ┌───────────────────┐         ┌───────────────────┐
    │ base_U: [d_out, r]│         │ lora_B: [d_out, k]│
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │ * scaling gamma
              └──────────────┬──────────────┘
                             │
                             ▼ (+) + bias b
                    Output Tensor y: [B, N, d_out]
```

---

## 4. API Reference & Factory Usage

### 4.1 ResSVDCompressor Class
```python
from framework.compression_engine.res_svd import ResSVDCompressor, SpectralResidualLoRA

compressor = ResSVDCompressor(
    model=model,
    base_rank=32,            # Rank r for frozen base
    residual_rank=8,         # Rank k for trainable residual adapter
    scaling=1.0,             # Adapter scaling factor alpha / k
    factorize_base=True,     # Store base as factorized (U, V)
    freeze_base=True,        # Set requires_grad=False on base
    target_modules=None,     # Layer filters
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Functions
```python
from framework.compression_engine.res_svd import (
    res_svd_factory,
    spectral_residual_lora_factory,
)

# Res-SVD factory
compressed_model = res_svd_factory(model, base_rank=32, residual_rank=8)

# Spectral Residual LoRA factory
peft_model = spectral_residual_lora_factory(model, base_rank=32, residual_rank=8)
```

### 4.3 Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | PyTorch neural network to compress. |
| `base_rank` | `int` | `32` | Rank $r$ of the frozen base low-rank matrix. |
| `residual_rank` | `int` | `8` | Rank $k$ of the trainable residual LoRA adapter. |
| `scaling` | `float` | `1.0` | Output scaling factor for the residual adapter. |
| `factorize_base` | `bool` | `True` | Whether base is factorized ($U, V$) or dense matrix. |
| `freeze_base` | `bool` | `True` | Whether base parameters have `requires_grad=False`. |

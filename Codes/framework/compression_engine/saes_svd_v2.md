# SAESSVDCompressorV2: Spectrum-Aware Error Suppression SVD (SAES-SVD v2)

## 1. Executive Abstract & Theoretical Motivation
Direct truncated SVD approximates a weight matrix $W$ by dropping trailing singular components $\sigma_{r+1}, \dots, \sigma_d$. In deep networks, the cumulative truncation errors across dozens of consecutive layers compound multiplicatively, causing representation degradation.

`SAESSVDCompressorV2` (registered as `saes_svd_v2` and `saes`) implements Spectrum-Aware Error Suppression (SAES). It initializes low-rank factors $A \in \mathbb{R}^{d_{\text{out}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{in}}}$ using symmetric square-root singular value allocation ($U \sqrt{\Sigma}, \sqrt{\Sigma} V^T$) and iteratively applies closed-form residual gradient corrections:

$$E = W - A B$$

updating $A$ and $B$ to actively minimize the residual error norm $\|E\|_F^2$ before replacing dense `nn.Linear` layers with factorized `SVDLinear` modules.

---

## 2. Mathematical Formulations

### 2.1 Initial Symmetric Factorization
Given weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$W = U \Sigma V^T$$

Truncating to rank $r$, the initial factor matrices are:

$$A^{(0)} = U[:, :r] \operatorname{diag}(\sqrt{\Sigma_{:r}}) \in \mathbb{R}^{d_{\text{out}} \times r}$$

$$B^{(0)} = \operatorname{diag}(\sqrt{\Sigma_{:r}}) V^T[:r, :] \in \mathbb{R}^{r \times d_{\text{in}}}$$

### 2.2 Iterative Residual Error Suppression
For each correction step $t = 1, \dots, T$:
1. **Compute Residual Error**:
   $$E^{(t)} = W - A^{(t-1)} B^{(t-1)}$$

2. **Update Left Factor $A$**:
   $$A^{(t)} = A^{(t-1)} + E^{(t)} (B^{(t-1)})^T \left(B^{(t-1)} (B^{(t-1)})^T + \epsilon I\right)^{-1}$$

3. **Update Right Factor $B$**:
   $$B^{(t)} = B^{(t-1)} + \left((A^{(t)})^T A^{(t)} + \epsilon I\right)^{-1} (A^{(t)})^T \left(W - A^{(t)} B^{(t-1)}\right)$$

This iterative refinement aligns the low-rank subspace with dominant cross-layer interaction dynamics and suppresses quantization/truncation error drift.

### 2.3 Structural Layer Replacement
The resulting refined factors $A^{(T)}$ and $B^{(T)}$ are loaded directly into `SVDLinear`:

$$y = (x B^T) A^T + b$$

---

## 3. Architecture & Execution Pipeline

```
            Dense Pretrained Weight W [d_out, d_in]
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Truncated SVD Initialization │
              │ A = U_r sqrt(S), B = sqrt(S)V_r^T │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Iterative Error Suppression │
              │ T steps of closed-form ALS  │
              │ E = W - AB                  │
              └──────────────┬──────────────┘
                             │  [Refined A, B]
                             ▼
              ┌─────────────────────────────┐
              │ SVDLinear Module Replacement│
              │ y = (x @ B.T) @ A.T + b     │
              └─────────────────────────────┘
```

---

## 4. API Reference & Factory Usage

### 4.1 Class Reference
```python
from framework.compression_engine.saes_svd_v2 import SAESSVDCompressorV2

compressor = SAESSVDCompressorV2(
    model=model,
    rank=32,                   # Target bottleneck rank
    rank_ratio=0.7,            # Fallback fraction if rank is None
    correcting_steps=3,        # Number of iterative error suppression steps
    target_modules=None,       # Layer filter
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Functions
```python
from framework.compression_engine.saes_svd_v2 import (
    saes_svd_v2_factory,
    saes_factory,
)

# Canonical factory invocation
compressed_model = saes_svd_v2_factory(
    model=model,
    comp_config=None,
    rank=32,
    correcting_steps=3,
)
```

### 4.3 Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | Neural network model to compress. |
| `rank` | `int` | `None` | Target low-rank dimension $r$. |
| `rank_ratio` | `float` | `0.7` | Retention ratio of maximum rank. |
| `correcting_steps`| `int` | `3` | Number of error suppression iterations. |
| `target_modules` | `List[str]` | `None` | Layer name substring filter. |

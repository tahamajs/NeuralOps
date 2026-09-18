# 5. Architecture and Implementation


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 SVDLinear Module
Dense layers selected for compression are substituted with `SVDLinear`:
$$y = (x V^\top) U^\top + \text{bias}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 74, 188-191 -->

When optional recovery adaptation is enabled, auxiliary LoRA parameters are evaluated in parallel:
$$y = (x V^\top) U^\top + \frac{\alpha}{r_{\text{lora}}} (x A_{\text{lora}}^\top) B_{\text{lora}}^\top + \text{bias}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 74 -->

## 5.2 Module Parameters
- `V`: `nn.Parameter` of shape $(r, d_{\text{in}})$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 105 -->
- `U`: `nn.Parameter` of shape $(d_{\text{out}}, r)$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 106 -->
- `bias`: `nn.Parameter` of shape $(d_{\text{out}})$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 108 -->
- `lora_A`: `nn.Parameter` of shape $(r_{\text{lora}}, d_{\text{in}})$ (optional) <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 111 -->
- `lora_B`: `nn.Parameter` of shape $(d_{\text{out}}, r_{\text{lora}})$ (optional) <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 112 -->


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 3.1 Dataflow & Activation Anchoring Pipeline

```
              Calibration Text Inputs
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Forward Pass with PyTorch Hooks              │
  │   Capture Input Activations X_l              │
  │   Shape: (B, N, d_in)                        │
  └──────────────────────┬───────────────────────┘
                         │  [Activation Tensors X]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation Matrix Flattening                 │
  │   X_flat = X.view(-1, d_in)  [Shape: (K, d_in)]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation SVD Decomposition                 │
  │   [U_X, S_X, Vh_X] = svd(X_flat)             │
  │   Anchor Basis: A_anchor = Vh_X[:r, :].T     │
  │   Shape: (d_in, r)                           │
  └──────────────────────┬───────────────────────┘
                         │  [Anchor Matrix A_anchor]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Subspace Projection of Layer Weight          │
  │   W_anchored = W @ A_anchor @ A_anchor.T     │
  │   Shape: (d_out, d_in)                       │
  └──────────────────────┬───────────────────────┘
                         │  [Anchored Weight W_anchored]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Anchored SVD Factorization & Reconstruction  │
  │   [U, S, Vt] = svd(W_anchored)               │
  │   W_approx = U[:, :r] @ diag(S[:r]) @ Vt[:r,:]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
              Compressed Model Layer Weight
```

### 3.2 Detailed Step-by-Step Trace
1. **Hook Registration**: Attach forward hooks to target `nn.Linear` layers in the evaluation network.
2. **Activation Harvesting**: Pass tokenized calibration texts ("The quick brown fox...") to generate internal feature maps. Intercept input tensor $X$.
3. **Activation Subspace Extraction**: Flatten $X$ into 2D matrix $\bar{X}$, execute PyTorch `linalg.svd`, extract top-$r$ right-singular vectors to construct $A_{\text{anchor}}$.
4. **Weight Projection & SVD**: Compute $W_{\text{anchored}} = W A_{\text{anchor}} A_{\text{anchor}}^T$, compute SVD, truncate to rank $r$, and assign reconstructed weight matrix $\hat{W}_{\text{AASVD}}$ to parameter storage.

---


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 3.1 Dataflow & Activation Anchoring Pipeline

```
              Calibration Text Inputs
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Forward Pass with PyTorch Hooks              │
  │   Capture Input Activations X_l              │
  │   Shape: (B, N, d_in)                        │
  └──────────────────────┬───────────────────────┘
                         │  [Activation Tensors X]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation Matrix Flattening                 │
  │   X_flat = X.view(-1, d_in)  [Shape: (K, d_in)]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation SVD Decomposition                 │
  │   [U_X, S_X, Vh_X] = svd(X_flat)             │
  │   Anchor Basis: A_anchor = Vh_X[:r, :].T     │
  │   Shape: (d_in, r)                           │
  └──────────────────────┬───────────────────────┘
                         │  [Anchor Matrix A_anchor]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Subspace Projection of Layer Weight          │
  │   W_anchored = W @ A_anchor @ A_anchor.T     │
  │   Shape: (d_out, d_in)                       │
  └──────────────────────┬───────────────────────┘
                         │  [Anchored Weight W_anchored]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Anchored SVD Factorization & Reconstruction  │
  │   [U, S, Vt] = svd(W_anchored)               │
  │   W_approx = U[:, :r] @ diag(S[:r]) @ Vt[:r,:]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
              Compressed Model Layer Weight
```

### 3.2 Detailed Step-by-Step Trace
1. **Hook Registration**: Attach forward hooks to target `nn.Linear` layers in the evaluation network.
2. **Activation Harvesting**: Pass tokenized calibration texts ("The quick brown fox...") to generate internal feature maps. Intercept input tensor $X$.
3. **Activation Subspace Extraction**: Flatten $X$ into 2D matrix $\bar{X}$, execute PyTorch `linalg.svd`, extract top-$r$ right-singular vectors to construct $A_{\text{anchor}}$.
4. **Weight Projection & SVD**: Compute $W_{\text{anchored}} = W A_{\text{anchor}} A_{\text{anchor}}^T$, compute SVD, truncate to rank $r$, and assign reconstructed weight matrix $\hat{W}_{\text{AASVD}}$ to parameter storage.

---


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 3.1 Dataflow & Activation Anchoring Pipeline

```
              Calibration Text Inputs
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Forward Pass with PyTorch Hooks              │
  │   Capture Input Activations X_l              │
  │   Shape: (B, N, d_in)                        │
  └──────────────────────┬───────────────────────┘
                         │  [Activation Tensors X]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation Matrix Flattening                 │
  │   X_flat = X.view(-1, d_in)  [Shape: (K, d_in)]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Activation SVD Decomposition                 │
  │   [U_X, S_X, Vh_X] = svd(X_flat)             │
  │   Anchor Basis: A_anchor = Vh_X[:r, :].T     │
  │   Shape: (d_in, r)                           │
  └──────────────────────┬───────────────────────┘
                         │  [Anchor Matrix A_anchor]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Subspace Projection of Layer Weight          │
  │   W_anchored = W @ A_anchor @ A_anchor.T     │
  │   Shape: (d_out, d_in)                       │
  └──────────────────────┬───────────────────────┘
                         │  [Anchored Weight W_anchored]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Anchored SVD Factorization & Reconstruction  │
  │   [U, S, Vt] = svd(W_anchored)               │
  │   W_approx = U[:, :r] @ diag(S[:r]) @ Vt[:r,:]│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
              Compressed Model Layer Weight
```

### 3.2 Detailed Step-by-Step Trace
1. **Hook Registration**: Attach forward hooks to target `nn.Linear` layers in the evaluation network.
2. **Activation Harvesting**: Pass tokenized calibration texts ("The quick brown fox...") to generate internal feature maps. Intercept input tensor $X$.
3. **Activation Subspace Extraction**: Flatten $X$ into 2D matrix $\bar{X}$, execute PyTorch `linalg.svd`, extract top-$r$ right-singular vectors to construct $A_{\text{anchor}}$.
4. **Weight Projection & SVD**: Compute $W_{\text{anchored}} = W A_{\text{anchor}} A_{\text{anchor}}^T$, compute SVD, truncate to rank $r$, and assign reconstructed weight matrix $\hat{W}_{\text{AASVD}}$ to parameter storage.

---

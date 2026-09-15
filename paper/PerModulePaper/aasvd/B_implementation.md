<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `SVDLinear` Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        
        # Base factors
        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
    
    def forward(self, x):
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out * self.output_scale
```

## B.2 Streaming EMA Calibration

```python
def make_hook(name):
    def hook(module, inp, outp):
        x = inp[0].detach()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_flat = x_fp32.reshape(-1, in_f)
            mask = x_flat.abs().sum(dim=-1) > 1e-5
            x_active = x_flat[mask]
            if x_active.size(0) == 0:
                return
            batch_cov = (
                torch.matmul(x_active.T, x_active) / max(1, x_active.size(0))
            ).cpu()
        
        if name not in covariances:
            covariances[name] = batch_cov
        else:
            covariances[name] = (
                self.ema_beta * covariances[name]
                + (1.0 - self.ema_beta) * batch_cov
            )
    return hook
```

**Key implementation notes:**

1. **FP32 compute:** We cast to FP32 inside the hook to avoid numerical issues.
2. **Padding filter:** Mask tokens with $\|x_i\|_1 < 10^{-5}$.
3. **CPU transfer:** `batch_cov.cpu()` prevents GPU memory accumulation.

## B.3 Eigendecomposition with Regularization

```python
def compute_sqrt_and_inv(cov, device):
    cov = cov.to(device=device, dtype=torch.float32)
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    
    # Regularize
    eps = 1e-6 * torch.max(eigvals).clamp(min=1e-12) + 1e-8
    eigvals_reg = eigvals.clamp(min=eps)
    
    sqrt_eigvals = torch.sqrt(eigvals_reg)
    inv_sqrt_eigvals = 1.0 / sqrt_eigvals
    
    cov_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    cov_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T
    
    return cov_sqrt, cov_inv_sqrt
```

## B.4 Weighted SVD

```python
def compress_linear(self, layer, cov_factors=None):
    W = weight(layer).float()
    
    if cov_factors is not None:
        cov_sqrt, cov_inv_sqrt = cov_factors
        W_weighted = W @ cov_sqrt
    else:
        W_weighted = W
    
    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
    
    # Truncate
    r = min(self.rank, ...)
    sqrt_S = torch.sqrt(S[:r].clamp(min=1e-12))
    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
    
    # Unweight
    if cov_inv_sqrt is not None:
        V_r = V_r @ cov_inv_sqrt
    
    # Validate
    W_rec = U_r @ V_r
    err = torch.norm(W - W_rec).item() / max(torch.norm(W).item(), 1e-12)
    
    return SVDLinear.from_factors(U_r, V_r), err, r
```

## B.5 Rank Cap Constraint

```python
max_allowed_rank = max(1, min(
    in_features,
    out_features,
    int(0.75 * (in_features * out_features) / (in_features + out_features))
))
r = min(r, max_allowed_rank)
```

This ensures $r(d_\text{in} + d_\text{out}) < 0.75 d_\text{in} d_\text{out}$, 
guaranteeing at least 25% parameter reduction per layer.

## B.6 Handling Conv1D

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    W = raw_w.t().float()  # (in, out) → (out, in)
else:
    W = raw_w.float()
```

## B.7 Common Pitfalls

### B.7.1 Forgetting CPU Transfer

Without `.cpu()`, GPU memory grows linearly with calibration batches.

### B.7.2 No Regularization

Without regularizing $\lambda_i$, near-zero eigenvalues produce inf in 
$C_X^{-1/2}$.

### B.7.3 Wrong Conv1D Orientation

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

### B.7.4 No Padding Filter

Without masking padding tokens, covariance is biased.

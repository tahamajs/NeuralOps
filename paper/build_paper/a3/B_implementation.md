<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code-level details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Module Structure

```python
class A3FactorizedLinear(nn.Module):
    """Factorized linear layer: y = (xV^T)U^T + b + LoRA"""
    
    def __init__(self, in_features, out_features, rank, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_rank = lora_rank
        
        # Base factors (frozen after copy)
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
        
        self.reset_parameters()
    
    def forward(self, x):
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out
```

## B.2 The Main Compression Loop

```python
def compress(self):
    # 1. Collect statistics
    cov_map, fisher_map = self._collect_activations_and_fisher()
    
    # 2. For each layer
    for name, module in self.model.named_modules():
        if not self._is_target(module, name):
            continue
        
        cov = cov_map.get(name)
        fisher = fisher_map.get(name)
        
        factorized, err, rank = self.compress_linear(
            module, name=name, cov=cov, fisher=fisher
        )
        
        if factorized is not None:
            # Replace
            if hasattr(module, 'base_layer'):
                module.base_layer = factorized
                replacement = module
            else:
                replacement = factorized
            replace_module(self.model, name, replacement)
        # else: fallback, keep original
    
    # 3. LN recalibration
    if self.recalibrate_ln:
        recalibrate_layernorms(self.model, ...)
    
    return self.model, self.get_stats()
```

## B.3 Streaming Statistics

```python
def _collect_activations_and_fisher(self):
    cov_running = {}
    fisher_running = {}
    count_running = {}
    
    def make_hook(name):
        def hook(module, inp, outp):
            with torch.no_grad():
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                mask = x.abs().sum(dim=-1) > 1e-5
                x_valid = x[mask]
                if x_valid.size(0) > 0:
                    c = torch.matmul(x_valid.T, x_valid)
                    f = x_valid.pow(2).sum(dim=0)
                    if name not in cov_running:
                        cov_running[name] = c.cpu()
                        fisher_running[name] = f.cpu()
                        count_running[name] = x_valid.size(0)
                    else:
                        cov_running[name] += c.cpu()
                        fisher_running[name] += f.cpu()
                        count_running[name] += x_valid.size(0)
        return hook
    
    hooks = []
    for name, module in self.model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration
    self.model.eval()
    with torch.no_grad():
        for batch in self.calibration_loader:
            self.model(**batch)
    
    for h in hooks:
        h.remove()
    
    # Normalize
    cov_map = {n: cov_running[n] / count_running[n] for n in cov_running}
    fisher_map = {n: fisher_running[n] / count_running[n] for n in fisher_running}
    
    return cov_map, fisher_map
```

## B.4 Weighted SVD

```python
def compress_linear(self, module, name, cov=None, fisher=None):
    W = module.weight.data.float()
    
    # Extract shape
    out_f, in_f = W.shape[0], W.shape[1]
    
    # Compute weighting
    if self.use_fisher and fisher is not None:
        scale = torch.sqrt(fisher.clamp(min=1e-8) + 1e-6)
    elif cov is not None:
        scale = torch.sqrt(torch.diag(cov).clamp(min=1e-8) + 1e-6)
    else:
        scale = None
    
    # Weighted matrix
    if scale is not None:
        if self.normalize_scale:
            rms = torch.norm(scale) / math.sqrt(in_f)
            scale = scale / (rms + 1e-8)
        scale = scale.clamp(min=1e-3, max=1e3)
        W_w = W * scale.unsqueeze(0)
    else:
        W_w = W
    
    # SVD
    U, S, Vt = torch.linalg.svd(W_w, full_matrices=False)
    
    # Retry loop
    norm_orig = torch.linalg.norm(W).item()
    best_err = float('inf')
    best_factors = None
    
    for r in adaptive_ranks:
        sqrt_S = torch.sqrt(S[:r].clamp(min=1e-12))
        U_r = U[:, :r] * sqrt_S.unsqueeze(0)
        V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
        if scale is not None:
            V_r = V_r / scale.unsqueeze(0)
        
        W_rec = U_r @ V_r
        err = torch.linalg.norm(W - W_rec).item() / (norm_orig + 1e-8)
        
        if err < best_err:
            best_err = err
            best_factors = (U_r, V_r)
        
        if err <= self.max_error:
            break
    
    if best_err > self.max_error and self.fallback_to_original:
        return None, best_err, 0
    
    return A3FactorizedLinear(
        in_features=in_f, out_features=out_f, rank=best_r,
        U_factor=best_factors[0], V_factor=best_factors[1]
    ), best_err, best_r
```

## B.5 LayerNorm Recalibration

```python
def recalibrate_layernorms(model, tokenizer, calibration_loader, ...):
    # 1. Record target std
    target_stds = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            def make_hook(n):
                def hook(mod, inp, outp):
                    x = inp[0].detach().float()
                    target_stds[n] = x.std(dim=-1).mean().item()
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Forward pass
    with torch.no_grad():
        for batch in calibration_loader:
            model(**batch)
    for h in hooks: h.remove()
    
    # 2. Compute current std (after compression)
    current_stds = {}
    # (same procedure)
    
    # 3. Rescale
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and name in target_stds:
            ratio = target_stds[name] / (current_stds[name] + 1e-8)
            with torch.no_grad():
                module.weight.data *= ratio
```

## B.6 Common Pitfalls

### B.6.1 Forgetting to Freeze Base Factors

```python
# RIGHT
self.V.requires_grad = False
self.U.requires_grad = False
# LoRA stays trainable
```

### B.6.2 Wrong Conv1D Orientation

GPT-2's `Conv1D` stores `(in, out)`, but `nn.Linear` stores `(out, in)`:

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    W = raw_w.t()  # transpose!
```

### B.6.3 Missing Padding Filter

Without filtering padding tokens, the Fisher estimate is biased:

```python
mask = x.abs().sum(dim=-1) > 1e-5
x_valid = x[mask]
```

### B.6.4 Numerical Overflow in Weighting

Clamp $\sqrt{F}$ to avoid extreme values:

```python
scale = scale.clamp(min=1e-3, max=1e3)
```

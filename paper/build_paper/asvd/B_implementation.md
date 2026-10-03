<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The ASVDLinear Module

```python
class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True,
                 lora_rank=0, freeze_base=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.S = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()
    
    def forward(self, x):
        Vx = F.linear(x, self.V)          # x @ V^T
        SVx = Vx * self.S                  # scale by Σ
        out = F.linear(SVx, self.U)        # @ U^T
        if self.bias is not None:
            out = out + self.bias
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out * self.output_scale
```

## B.2 Energy-Based Rank Selection

```python
def _decompose_and_set(self, target, src_weight, src_bias, proj_name):
    W = src_weight.detach().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    # Cumulative energy
    total_energy = torch.sum(S ** 2)
    cum_energy = torch.cumsum(S ** 2, dim=0) / (total_energy + 1e-12)
    
    # Select rank
    target_energy = max(self.energy_threshold, 
                       1.0 - self.error_threshold ** 2)
    indices = (cum_energy >= target_energy).nonzero()
    r = indices[0].item() + 1 if len(indices) > 0 else min(U.size(1), Vt.size(0))
    
    if self.dynamic_rank:
        r = max(16, min(r, U.size(1), Vt.size(0)))
    else:
        r = min(self.rank, U.size(1), Vt.size(0))
    
    # Validate
    w_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
    rel_err = float((torch.norm(W - w_rec) / (torch.norm(W) + 1e-9)).item())
    
    if rel_err > self.error_threshold:
        raise RuntimeError(...)
    
    target.set_svd_factors(U[:, :r], Vt[:r, :], S[:r], bias=src_bias)
    return rel_err
```

## B.3 The `set_identity` Method

```python
def set_identity(self):
    with torch.no_grad():
        self.U.data.zero_()
        self.V.data.zero_()
        self.S.data.fill_(1.0)
        r = min(self.rank, self.in_features, self.out_features)
        self.U.data[:r, :r].fill_diagonal_(1.0)
        self.V.data[:r, :r].fill_diagonal_(1.0)
        if self.bias is not None:
            self.bias.data.zero_()
        self.freeze_base_weights()
```

## B.4 Pre-Flight Extraction

In `inject_asvd`:

```python
try:
    new_attn = ASVDAttention(target_config, rank=rank,
                              pretrained_attn=old_attn, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
except RuntimeError as err:
    logger.warning(f"Pre-flight extraction failed: {err}. Aborting.")
```

## B.5 Common Pitfalls

### B.5.1 Forgetting `set_identity` for Nested Encoders

Without `set_identity`, RoBERTa double-projects the output.

### B.5.2 Wrong Conv1D Transpose

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

### B.5.3 Missing `freeze_base_weights`

Without freezing, recovery training modifies the base factors and undoes 
compression.

### B.5.4 Not Handling `output_scale`

The `output_scale` parameter is initialized to 1.0 but unused. Removing 
it would simplify the module.

---
# Appendix B. Implementation Details

## B.1 Core Module

```python
class SOAFusedAttention(BaseAttention):
    def __init__(self, config, k_rank=64, ...):
        super().__init__(config, rank=k_rank)
        
        self.dim = ...
        self.k_rank = max(128, self.dim // 4) if k_rank == 64 else k_rank
        
        # Down-projection (frozen)
        self.q_down = nn.Linear(self.dim, self.k_rank, bias=False)
        self.k_down = nn.Linear(self.dim, self.k_rank, bias=False)
        self.v_down = nn.Linear(self.dim, self.k_rank, bias=False)
        
        # Up-projection with LoRA
        self.q_up = FactorizedLoRAUp(self.k_rank, self.all_head_size, 
                                      lora_rank=lora_rank)
        self.k_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, 
                                      lora_rank=lora_rank)
        self.v_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, 
                                      lora_rank=lora_rank)
        
        # Output (not factorized)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, 
                                    lora_rank=lora_rank)
```

## B.2 SVD Initialization

```python
with torch.no_grad():
    for proj_key, down, up, target_out in [
        ("q", self.q_down, self.q_up, self.all_head_size),
        ("k", self.k_down, self.k_up, self.all_kv_head_size),
        ("v", self.v_down, self.v_up, self.all_kv_head_size),
    ]:
        w, _ = extracted[proj_key]
        w_f = w.float()
        
        U, S, Vt = torch.linalg.svd(w_f, full_matrices=False)
        S = S.clamp(min=1e-6)
        r = min(self.k_rank, U.size(1), Vt.size(0))
        
        down.weight.data.zero_()
        up.weight.data.zero_()
        down.weight.data[:r, :].copy_(Vt[:r, :])
        up.weight.data[:, :r].copy_(U[:, :r] * S[:r])
```

## B.3 Freezing

```python
if self.freeze_base:
    # Down projections always frozen
    for mod in [self.q_down, self.k_down, self.v_down]:
        mod.weight.requires_grad = False
    
    # Up projections: freeze base, keep LoRA trainable
    for mod in [self.q_up, self.k_up, self.v_up]:
        mod.freeze_base_weights()
    
    # Output projection
    self.out_proj.freeze_base_weights()
```

## B.4 Nested Encoder Bypass

```python
if self.is_nested_encoder:
    if not any(p.requires_grad for p in self.parameters()):
        raise RuntimeError("No trainable parameters — backward will fail")
    return context, (probs if output_attentions else None)
out = self.out_proj(context)
return out, (probs if output_attentions else None)
```

## B.5 Failure Handling

```python
missing_after = [k for k, (w, _) in extracted.items() 
                 if w is None and (k != "o" or not self.is_nested_encoder)]
if missing_after:
    raise RuntimeError(
        f"SOAFusedAttention copy_weights failed: missing {missing_after}"
    )
```

## B.6 Frobenius Validation

```python
fnorm = torch.linalg.norm(torch.matmul(up.weight, down.weight).float()).item()
if fnorm < 1e-4:
    raise RuntimeError(f"Weight copy failed for {proj_key}: norm {fnorm}")
```

## B.7 Common Pitfalls

### B.7.1 Forgetting to Transpose Conv1D

GPT-2 Conv1D stores `(in, out)`. Must transpose.

### B.7.2 Wrong Nested Detection

For RoBERTa, `is_nested_encoder=True` and `out_proj` is identity.

### B.7.3 LoRA on Wrong Factor

LoRA should be on the **up-projection** (after down). Placing it on down 
would increase parameter count without benefit.

### B.7.4 Missing Frobenius Check

Without the check, a silent failure can produce a random-init projection.

## B.8 The `FactorizedLoRAUp` Module

```python
class FactorizedLoRAUp(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=16, lora_alpha=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / lora_rank
        
        self.weight.requires_grad = False  # Always frozen
    
    def forward(self, x):
        out = F.linear(x, self.weight)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out
```
```

---
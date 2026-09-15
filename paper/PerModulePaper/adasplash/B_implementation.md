<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `AdaSplashLinear` Module

```python
class AdaSplashLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = min(lora_rank, in_features, out_features)
        
        # Frozen base
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features),
                requires_grad=False
            )
        
        # Trainable LoRA
        if self.lora_rank > 0:
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features))
        
        self.reset_parameters()
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out
```

## B.2 Entmax Bisection

```python
def entmax_bisect(scores, alpha=1.5, valid_mask=None, dim=-1, n_iter=25):
    device = scores.device
    dtype = scores.dtype
    neg_inf_val = torch.tensor(-10000.0, device=device, dtype=dtype)
    
    # Ensure finite
    scores = torch.where(torch.isfinite(scores), scores, neg_inf_val)
    
    # Softmax fallback for alpha = 1
    if abs(alpha - 1.0) < 1e-4:
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)
        sf = torch.nan_to_num(F.softmax(scores.float(), dim=dim), nan=0.0).to(dtype)
        if valid_mask is not None:
            sf = sf.masked_fill(~valid_mask, 0.0)
        sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
        return (sf / sf_denom).to(dtype)
    
    alpha_minus_1 = max(1e-4, alpha - 1.0)
    inv_alpha_minus_1 = 1.0 / alpha_minus_1
    
    # Mask invalid keys
    if valid_mask is not None:
        scores_valid = scores.masked_fill(~valid_mask, -10000.0)
    else:
        scores_valid = scores
    
    # Shift scores
    max_scores = scores_valid.amax(dim=dim, keepdim=True)
    max_scores = torch.where(max_scores < -5000.0, torch.zeros_like(max_scores), max_scores)
    scores_shift = scores_valid - max_scores
    scaled = scores_shift * alpha_minus_1
    
    # Initial bounds
    if valid_mask is not None:
        scaled_for_min = scaled.masked_fill(~valid_mask, 10000.0)
        tau_lo = scaled_for_min.amin(dim=dim, keepdim=True) - 1.0
        tau_lo = torch.where(tau_lo > 5000.0, torch.zeros_like(tau_lo) - 1.0, tau_lo)
    else:
        tau_lo = scaled.amin(dim=dim, keepdim=True) - 1.0
    tau_hi = torch.zeros_like(tau_lo)
    
    # Bisection
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        sum_p = p.sum(dim=dim, keepdim=True)
        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)
    
    # Final
    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)
    
    denom = p.sum(dim=dim, keepdim=True)
    
    # Softmax fallback
    scores_for_sm = scores.masked_fill(~valid_mask, -10000.0) if valid_mask is not None else scores
    sf = torch.nan_to_num(F.softmax(scores_for_sm.float(), dim=dim), nan=0.0).to(dtype)
    if valid_mask is not None:
        sf = sf.masked_fill(~valid_mask, 0.0)
    sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
    sf = sf / sf_denom
    
    is_empty = (denom < 1e-7) | torch.isnan(denom)
    p_normalized = torch.nan_to_num(p / denom.clamp(min=1e-9), nan=0.0)
    p = torch.where(is_empty, sf, p_normalized)
    return torch.nan_to_num(p, nan=0.0)
```

## B.3 Splash Mask Construction

```python
if is_decoder and splash_radius > 0 and not is_cross_attention and L_k > splash_radius:
    q_idx = torch.arange(L_q, device=device).unsqueeze(1)
    k_idx = torch.arange(L_k, device=device).unsqueeze(0)
    dist = q_idx - k_idx
    splash_local = ((dist >= 0) & (dist <= splash_radius)) | (k_idx < 4)
    splash_mask_4d = splash_local.unsqueeze(0).unsqueeze(0)
    logits = logits.masked_fill(~splash_mask_4d, -10000.0)
```

## B.4 Weight Extraction

```python
def copy_weights(self, src):
    from framework.core.adapters import UniversalAdapter
    
    extracted_weights = {}
    
    for proj in ["q", "k", "v", "o"]:
        w, b = None, None
        
        # Try UniversalAdapter
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
        except Exception:
            pass
        
        # Fallbacks
        if w is None:
            if proj == "o":
                # Check RoBERTa/BERT outer, GPT-2 c_proj, DistilBERT out_lin
                if hasattr(src, "output") and hasattr(src.output, "dense"):
                    w = src.output.dense.weight
                    b = getattr(src.output.dense, "bias", None)
                elif hasattr(src, "c_proj"):
                    w = src.c_proj.weight.T
                    b = getattr(src.c_proj, "bias", None)
                elif hasattr(src, "out_lin"):
                    w = src.out_lin.weight
                    b = getattr(src.out_lin, "bias", None)
            elif proj in ["q", "k", "v"] and hasattr(src, "c_attn"):
                fused = src.c_attn.weight.T
                split = fused.size(0) // 3
                idx = {"q": 0, "k": 1, "v": 2}[proj]
                w = fused[idx*split:(idx+1)*split]
                b = src.c_attn.bias[idx*split:(idx+1)*split] if src.c_attn.bias is not None else None
            elif hasattr(src, "self") and hasattr(src.self, "query"):
                s = src.self
                name_map = {"q": "query", "k": "key", "v": "value"}
                w = getattr(s, name_map[proj]).weight
                b = getattr(getattr(s, name_map[proj]), "bias", None)
            # ... more cases
        
        extracted_weights[proj] = (w, b)
    
    # Copy into target projections
    for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
        w, b = extracted_weights[proj]
        if w is not None:
            with torch.no_grad():
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
                # Freeze base, init LoRA
                target.weight.requires_grad = False
                if target.bias is not None:
                    target.bias.requires_grad = False
                if target.lora_A is not None:
                    nn.init.kaiming_uniform_(target.lora_A, a=math.sqrt(5))
                    target.lora_A.requires_grad = True
                if target.lora_B is not None:
                    nn.init.zeros_(target.lora_B)
                    target.lora_B.requires_grad = True
```

## B.5 Common Pitfalls

### B.5.1 Support Empty

If bisection fails to find valid τ, softmax fallback is essential.

### B.5.2 NaN in Scores

Mask to -10000 before bisection.

### B.5.3 Wrong Mask for Encoders

Splash masking is **only** for causal decoders. Encoders use full attention.

### B.5.4 Conv1D Transpose

GPT-2 Conv1D stores `(in, out)`. Must transpose.

### B.5.5 Attention Sinks

Preserving first 4 tokens is not optional — accuracy drops without it.

## B.6 Diagnostics

```python
from framework.attentions.adasplash import AdaSplashAttention, entmax_bisect

# Test entmax
scores = torch.randn(2, 4, 8, 8)
p = entmax_bisect(scores, alpha=1.5)
sparsity = (p == 0).float().mean()
print(f"Sparsity: {sparsity:.2%}")

# Check attention
for name, mod in model.named_modules():
    if isinstance(mod, AdaSplashAttention):
        print(name, mod.alpha, mod.splash_radius)
```

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `FactorisedLinear` Module

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, bias=True,
                 energy=0.95, lora_rank=16, lora_alpha=32.0, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
    
    def forward(self, x):
        if self.down.weight.device != x.device:
            self.to(device=x.device)
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A.to(x.dtype)), 
                            self.lora_B.to(x.dtype)) * self.lora_scaling
        return out
```

## B.2 `decompose_from_weight`

```python
def decompose_from_weight(self, weight, bias=None, energy=None):
    with torch.no_grad():
        W = weight.float()

        # Handle GQA dimension mismatch
        if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
            rep = self.out_features // W.shape[0]
            W = W.repeat(rep, 1)
        elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
            rep = W.shape[0] // self.out_features
            W = W.view(self.out_features, rep, -1).mean(dim=1)

        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Energy-based rank
        target_energy = energy if energy is not None else self.energy
        total_energy = (S**2).sum()
        cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)
        calc_rank = int((cum_energy < target_energy).sum().item()) + 1
        r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
        r = max(1, r)

        # Symmetric distribution
        sqrt_s = torch.sqrt(S[:r])
        A = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
        B = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(self.down.weight.dtype)

        # Store
        self.up.weight.data.zero_()
        self.down.weight.data.zero_()
        self.up.weight.data[:, :r] = A
        self.down.weight.data[:r, :] = B

        self.freeze_base_weights()
```

## B.3 Injection Hook

```python
def inject_minimal_prism(layer=None, model=None, ...):
    target = layer if layer is not None else model
    
    # 1. Whole model traversal
    if hasattr(target, "distilbert"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    
    if layer_list is not None:
        # Pre-test on first layer
        test_src = layer_list[0].attention
        try:
            MinimalPRISMAttention(cfg).copy_weights(test_src)
        except Exception as e:
            logger.warning(f"Pre-test failed: {e}")
            return None
        
        # Inject all
        for idx, lyr in enumerate(layer_list):
            inject_minimal_prism(layer=lyr, ...)
        return target
    
    # 2. Single-layer injection
    src_attn = get_attention(target)
    new_attn = MinimalPRISMAttention(cfg)
    new_attn.copy_weights(src_attn)
    
    # Set nested flag at injection
    if hasattr(target, "attention") and hasattr(target.attention, "self"):
        new_attn.is_nested_encoder = True
        target.attention.self = new_attn
    else:
        new_attn.is_nested_encoder = False
        if hasattr(target, "attention"):
            target.attention = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        ...
    
    return new_attn
```

## B.4 Handling GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    c_attn_mod = src.c_attn
    base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
    weight = base_c.weight.data.clone()

    is_conv1d = (
        "Conv1D" in type(c_attn_mod).__name__
        or "Conv1D" in type(base_c).__name__
        or hasattr(c_attn_mod, "nf")
        or hasattr(base_c, "nf")
    )
    if is_conv1d:
        weight = weight.t()

    split_size = weight.shape[0] // 3
    q_w = weight[:split_size, :]
    k_w = weight[split_size : 2 * split_size, :]
    v_w = weight[2 * split_size :, :]
```

## B.5 Nested Encoder Detection

```python
cls_name = src.__class__.__name__
has_out_proj = any(
    hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj")
)

is_nested = False
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention") or (
    cls_name.endswith("SelfAttention") and not has_out_proj
):
    is_nested = True

self.is_nested_encoder = is_nested
```

## B.6 Common Pitfalls

### B.6.1 Conv1D Not Transposed

```python
# WRONG
w = src.c_attn.weight

# RIGHT
if is_conv1d:
    w = w.t()
```

### B.6.2 Nested Encoder Wrong Flag

```python
# WRONG (RoBERTa)
self.is_nested_encoder = False  # out_proj applied twice

# RIGHT
self.is_nested_encoder = True
```

### B.6.3 LoRA B Not Zeroed

```python
# WRONG
self.lora_B = nn.Parameter(torch.randn(...))

# RIGHT
self.lora_B = nn.Parameter(torch.zeros(...))
```

### B.6.4 Missing Near-Zero Check

```python
w_norm = torch.linalg.norm(w.float())
if w_norm < 1e-4:
    raise RuntimeError(f"Near-zero weight on {proj}")
```

## B.7 Injection Safety Protocol

```python
# Pre-test on first layer
test_src = layer_list[0].attention
test_module = MinimalPRISMAttention(cfg)
try:
    test_module.copy_weights(test_src)
except Exception as e:
    logger.warning(f"Pre-test failed: {e}. Aborting.")
    return None
```

## B.8 Double-Injection Guard

```python
if isinstance(src_attn, MinimalPRISMAttention) or "MinimalPRISM" in src_attn.__class__.__name__:
    return None
```

---
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
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out
```

## B.2 The `decompose_from_weight` Method

```python
def decompose_from_weight(self, weight, bias=None, energy=None):
    with torch.no_grad():
        W = weight.float()
        
        # Handle KV dimension matching for GQA
        if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
            rep = self.out_features // W.shape[0]
            W = W.repeat(rep, 1)
        elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
            rep = W.shape[0] // self.out_features
            W = W.view(self.out_features, rep, -1).mean(dim=1)
        
        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Energy-based rank selection
        target_energy = energy if energy is not None else self.energy
        total_energy = (S ** 2).sum()
        cum_energy = torch.cumsum(S ** 2, dim=0) / (total_energy + 1e-12)
        calc_rank = int((cum_energy < target_energy).sum().item()) + 1
        r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
        r = max(1, r)
        
        # Symmetric factorization
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

## B.3 The Main Injection Loop

```python
def inject_minimal_nova(layer=None, model=None, ...):
    target = layer if layer is not None else model
    
    # Whole model traversal
    if hasattr(target, "distilbert"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer"):
        layer_list = target.transformer.h
    # ...
    
    if layer_list is not None:
        # Pre-test on first layer
        test_src = layer_list[0].attention
        test_module = MinimalNOVAAttention(config=cfg)
        test_module.copy_weights(test_src)
        
        # Inject all layers
        for idx, lyr in enumerate(layer_list):
            inject_minimal_nova(layer=lyr, ...)
        return target
    
    # Single layer injection
    src_attn = get_attention(target)
    new_attn = MinimalNOVAAttention(config=cfg)
    new_attn.copy_weights(src_attn)
    replace_attention(target, new_attn)
    return new_attn
```

## B.4 The `copy_weights` Method

Handles four architecture families:

```python
def copy_weights(self, src):
    extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}
    
    # 1. UniversalAdapter
    for proj in ["q", "k", "v", "o"]:
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
            if w is not None:
                extracted[proj] = (w, b)
        except Exception as e:
            logger.debug(f"UA failed for {proj}: {e}")
    
    # 2. GPT-2 fused c_attn
    if hasattr(src, "c_attn") and extracted["q"][0] is None:
        # ... split logic
    
    # 3. RoBERTa nested
    if hasattr(src, "self") and extracted["q"][0] is None:
        # ... extract from self.query, self.key, self.value
    
    # 4. Modern decoders (Qwen, DistilBERT)
    for proj, names in named_map.items():
        # ... extract from q_proj, q_lin, query
    
    # Detect nested encoder
    is_nested = ...
    self.is_nested_encoder = is_nested
    
    # Factorize each projection
    for proj_key, target in [("q", self.q_proj), ...]:
        w, b = extracted[proj_key]
        if w is not None:
            target.decompose_from_weight(w, b, self.energy)
        elif proj_key == "o" and self.is_nested_encoder:
            eye = torch.eye(...)
            target.decompose_from_weight(eye, None, self.energy)
```

## B.5 Handling GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    c_attn_mod = src.c_attn
    base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
    weight = base_c.weight.data.clone()
    
    # Conv1D stores (in, out)
    is_conv1d = (
        "Conv1D" in type(c_attn_mod).__name__
        or "Conv1D" in type(base_c).__name__
        or hasattr(c_attn_mod, "nf")
        or hasattr(base_c, "nf")
    )
    if is_conv1d:
        weight = weight.t()   # (out, in)
    
    split_size = weight.shape[0] // 3
    q_w = weight[:split_size, :]
    k_w = weight[split_size:2*split_size, :]
    v_w = weight[2*split_size:, :]
```

## B.6 Nested Encoder Bypass

```python
# In forward:
if self.is_nested_encoder:
    return context, attn_weights   # No out_proj
else:
    output = self.out_proj(context)
    return output, attn_weights
```

## B.7 Common Pitfalls

### B.7.1 Conv1D Not Transposed

```python
# WRONG
w = src.c_attn.weight    # shape (in, out) — wrong orientation

# RIGHT
if is_conv1d:
    w = w.t()    # (out, in)
```

### B.7.2 Nested Encoder Wrong Flag

```python
# WRONG (RoBERTa)
self.is_nested_encoder = False   # out_proj applied twice → collapse

# RIGHT
self.is_nested_encoder = True
```

### B.7.3 Missing Near-Zero Check

```python
w_norm = torch.linalg.norm(w.float())
if w_norm < 1e-4:
    raise RuntimeError(f"Near-zero weight on {proj}")
```

### B.7.4 LoRA B Not Zeroed

```python
# WRONG
self.lora_B = nn.Parameter(torch.randn(...))

# RIGHT
self.lora_B = nn.Parameter(torch.zeros(...))   # B = 0 at init
```

---
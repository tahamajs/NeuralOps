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
                 energy=0.95, lora_rank=16, lora_alpha=32.0):
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
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
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
        total_energy = (S ** 2).sum()
        cum_energy = torch.cumsum(S ** 2, dim=0) / (total_energy + 1e-12)
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

## B.3 The Injection Hook (Minimal)

```python
def inject_minimal_svd(layer, model_config, comp_config=None,
                       layer_idx=0, total_layers=1, old_attn=None, **kwargs):
    if model_config is None:
        model_config = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()
    
    energy = getattr(comp_config, "svd_energy", 0.95) if comp_config else 0.95
    dropout = getattr(comp_config, "dropout", 0.1) if comp_config else 0.1
    
    new_attn = MinimalSVDAttention(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
    )
    
    src_attn = old_attn if old_attn is not None else \
               getattr(layer, "attention", getattr(layer, "self", None))
    
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error(f"copy_weights failed: {e}")
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
            new_attn.is_nested_encoder = False
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    else:
        new_attn.is_nested_encoder = False
    
    return new_attn
```

**Note.** This function is intentionally minimal:

- **No whole-model traversal** — caller handles
- **No pre-test** — simplicity over safety
- **No double-injection guard** — assume well-formed caller

## B.4 Whole-Model Traversal (Caller Responsibility)

The caller must traverse the model:

```python
model = DistilBertForSequenceClassification.from_pretrained("...")

# DistilBERT
for layer in model.distilbert.transformer.layer:
    inject_minimal_svd(layer, model.config, energy=0.95)

# RoBERTa
for layer in model.roberta.encoder.layer:
    inject_minimal_svd(layer, model.config, energy=0.95)

# GPT-2
for layer in model.transformer.h:
    inject_minimal_svd(layer, model.config, energy=0.95)
```

## B.5 Handling GPT-2 Conv1D

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
    k_w = weight[split_size:2*split_size, :]
    v_w = weight[2*split_size:, :]
```

## B.6 Nested Encoder Detection

MinimalSVD sets `is_nested_encoder` in **two places**:

**In `__init__`:**

```python
model_type = getattr(cfg, "model_type", "").lower() if cfg else ""
self.is_nested_encoder = (
    getattr(cfg, "is_nested_encoder", False)
    or model_type in ("roberta", "bert", "xlm-roberta")
)
```

**In `copy_weights`:**

```python
cls_name = src.__class__.__name__
has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj"))

is_nested = False
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif (
    cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
    or (cls_name.endswith("SelfAttention") and not has_out_proj)
):
    is_nested = True

self.is_nested_encoder = is_nested
```

**Note.** `copy_weights` overrides the `__init__` value. This is important 
because `__init__` only has the config, while `copy_weights` inspects the 
actual module.

## B.7 Common Pitfalls

### B.7.1 Conv1D Not Transposed

```python
# WRONG
w = src.c_attn.weight

# RIGHT
if is_conv1d:
    w = w.t()
```

### B.7.2 Nested Encoder Wrong Flag

```python
# WRONG (RoBERTa)
self.is_nested_encoder = False   # out_proj applied twice

# RIGHT
self.is_nested_encoder = True
```

### B.7.3 LoRA B Not Zeroed

```python
# WRONG
self.lora_B = nn.Parameter(torch.randn(...))

# RIGHT
self.lora_B = nn.Parameter(torch.zeros(...))
```

### B.7.4 Missing Near-Zero Check

```python
w_norm = torch.linalg.norm(w.float())
if w_norm < 1e-4:
    raise RuntimeError(f"Near-zero weight on {proj}")
```

## B.8 Comparing with Companion Variants

| Feature | MinimalSVD | MinimalNOVA | MinimalPRISM |
|---|---|---|---|
| Whole-model traversal | ❌ | ✅ | ✅ |
| Pre-test | ❌ | ✅ | ✅ |
| Injection-time nested detection | ❌ | ❌ | ✅ |
| Double-injection guard | ❌ | ❌ | ✅ |
| Line count | ~280 | ~500 | ~550 |

---
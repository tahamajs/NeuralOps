<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `DobiSVDLinearLayer`

```python
class DobiSVDLinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        init_gate_val=2.5,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.V = nn.Parameter(torch.empty(self.rank, in_features, ...))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, ...))
        self.gate = nn.Parameter(torch.full((self.rank,), init_gate_val, ...))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, ...))

        self.reset_parameters()

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            # Pad or truncate
            ...
        g = torch.sigmoid(self.gate).unsqueeze(1)  # (rank, 1)
        V_gated = self.V * g
        latent = F.linear(x, V_gated)
        out = F.linear(latent, self.U, self.bias)
        return out
```

## B.2 Initialization with Gate Compensation

```python
# In compress_linear:
init_gate_scale = 1.0 / (1.0 / (1.0 + math.exp(-self.init_gate_val)))

U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :] * init_gate_scale).to(dtype=dtype, device=device)

dobi_layer = DobiSVDLinearLayer(
    in_features=in_f,
    out_features=out_f,
    rank=r,
    bias=target_mod.bias is not None,
    init_gate_val=self.init_gate_val,
    dtype=dtype,
    device=device,
)
dobi_layer.set_low_rank_factors(U_r, V_r)
```

## B.3 Reconstruction Validation

```python
W_recon = dobi_layer.weight.float()  # This uses sigmoid(gate) internally
err = float(torch.norm(W - W_recon, p="fro").item() / norm_w)
```

The `.weight` property computes:

```python
@property
def weight(self):
    g = torch.sigmoid(self.gate).unsqueeze(1)
    return torch.matmul(self.U, self.V * g)
```

## B.4 The Compressor

```python
class DobiSVDCompressor(BaseCompressor):
    def compress(self):
        for name, module in target_layers:
            new_layer, err, r = self.compress_linear(module)
            
            if new_layer is None or err > self.max_error:
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue
            
            # Replace
            if hasattr(module, "base_layer"):
                module.base_layer = new_layer
            elif hasattr(module, "linear") and hasattr(module, "lora_A"):
                module.linear = new_layer
            else:
                replace_module(self.model, name, new_layer)
```

## B.5 Registry

```python
@COMPRESSOR_REGISTRY.register("dobi")
@COMPRESSOR_REGISTRY.register("dobi_svd")
@COMPRESSOR_REGISTRY.register("dobi_compressor")
def dobi_factory(model, comp_config, **kwargs): ...
```

## B.6 Post-Hoc Pruning Script

```python
def prune_gates(model, tau=0.5):
    for name, module in model.named_modules():
        if isinstance(module, DobiSVDLinearLayer):
            g = torch.sigmoid(module.gate)
            keep = g > tau
            r_eff = keep.sum().item()
            
            if r_eff < module.rank:
                # Physically prune
                module.V.data = module.V.data[keep, :]
                module.U.data = module.U.data[:, keep]
                module.gate.data = module.gate.data[keep]
                module.rank = r_eff
```

## B.7 Common Pitfalls

### B.7.1 Forgetting Gate Compensation

Without dividing $V$ by $\sigma(\alpha_0)$:
- Initial forward is **scaled** by $\sigma(\alpha_0)$
- Model starts with reduced magnitude
- Slower convergence

### B.7.2 Wrong Init Gate Value

Too small ($\alpha_0 < 1$): initial forward is significantly scaled down.
Too large ($\alpha_0 > 5$): gates saturate, gradients vanish.

Default $\alpha_0 = 2.5$ is a good balance.

### B.7.3 Conv1D Orientation

```python
if "conv1d" in cls_name or hasattr(target_mod, "nf"):
    # Conv1D stores (in, out); need (out, in)
    W = raw_w.t().float()
```

### B.7.4 Missing LoRA Preservation

When a layer is wrapped in `LoRALinear`:

```python
if hasattr(module, "base_layer"):
    module.base_layer = new_layer  # Preserve wrapper
    # LoRA stays trainable
    for p_name, p in module.named_parameters():
        if "lora_" in p_name.lower():
            p.requires_grad = True
```

## B.8 Gate Dynamics Monitoring

During training, monitor gate statistics:

```python
for name, module in model.named_modules():
    if isinstance(module, DobiSVDLinearLayer):
        gates = torch.sigmoid(module.gate)
        print(f"{name}: mean={gates.mean():.3f} min={gates.min():.3f}")
```

## B.9 Fallback Behavior

When `fallback_to_original = True` and error exceeds threshold:

```python
self.fallback_layers.append(name)
for p in module.parameters():
    p.requires_grad = False  # Freeze the original layer
```

This ensures recovery training does not touch fallback layers.

## B.10 Comparison with Companion Methods

| Feature | MinimalSVD | SVD-LLM V2 | Dobi-SVD |
|---|---|---|---|
| Rank selection | Energy | Energy | **Learnable gates** |
| Gate parameter | ❌ | ❌ | **✅ (per rank)** |
| Differentiable | ❌ | ❌ | **✅** |
| Post-hoc pruning | ❌ | ❌ | **✅** |
| Gate compensation | ❌ | ❌ | **✅** |
| Code lines | ~280 | ~600 | ~450 |

---
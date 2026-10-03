<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Long-Memory Branch

```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**This is the complete implementation.** Despite the name, there is no 
gating, no delta rule, no recurrence.

**Recommended replacement (not implemented):**

```python
class RealGatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, 1)
        # ...

    def forward(self, x):
        # Real delta-rule recurrence
        # ...
```

## B.2 The Gate and Mean-Pooling

```python
self.gate = nn.Linear(self.dim, self.num_heads)
self.gate.bias.data.fill_(gate_init_bias)  # default 1.8
# ...
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)  # (B, L, 1)
```

**The `mean(dim=-1)`** collapses the head dimension. All heads receive the 
same gate value.

**Alternative (per-head gating):**

```python
gate = torch.sigmoid(self.gate(x))  # (B, L, H)
gate = gate.transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
short_per_head = short_out.view(B, H, L, self.head_dim)
long_per_head = long_out.view(B, H, L, self.head_dim)
fused = gate * short_per_head + (1 - gate) * long_per_head
```

## B.3 The Preservation Gate

```python
self.output_scale = nn.Parameter(torch.ones(1))
self.output_gate = nn.Parameter(torch.tensor(2.0))
# ...
out = self.out_proj(fused)
out = out * self.output_scale
alpha = torch.sigmoid(self.output_gate)
out = alpha * out + (1.0 - alpha) * x
```

**Concern.** `out_proj` is `LoRALinear` which itself has `output_scale`. 
This creates **double scaling**.

## B.4 Weight Transfer

Standard four-architecture handling:

```python
if hasattr(src, "c_attn"):
    # GPT-2 fused Conv1D
    ...
elif hasattr(src, "self") and hasattr(src.self, "query"):
    # RoBERTa nested
    ...
else:
    # DistilBERT flat
    ...
```

## B.5 Common Pitfalls

### B.5.1 The Fake Long-Memory Branch

The `GatedDeltaNetLayer` is a linear projection. It does not provide long-range 
memory.

### B.5.2 Double `output_scale`

`LoRALinear.forward` ends with `return out * self.output_scale`. 
`GoldFinchAHNBlock.forward` also multiplies by `self.output_scale`. This 
can cause numerical issues.

### B.5.3 Gate Initialization

`gate.bias.data.fill_(1.8)` gives $\sigma(1.8) \approx 0.86$. This means 
the model initially favors sliding-window attention by 86%.

### B.5.4 Missing `output_scale` in LoRALinear

The `output_scale` parameter in `LoRALinear` is a **new parameter** not 
present in the pretrained model. On `load_state_dict`, it defaults to 1.0.

## B.6 Diagnostics

### Check Gate Value

```python
x = torch.randn(2, 8, 768)
gate_raw = torch.sigmoid(model.gate(x))  # (2, 8, 12)
gate = gate_raw.mean(dim=-1, keepdim=True)  # (2, 8, 1)
print(f"Gate mean: {gate.mean().item():.4f}")
# At init, should be ≈ 0.86
```

### Check Preservation Gate

```python
alpha = torch.sigmoid(model.output_gate)
print(f"α = {alpha.item():.4f}")
# At init: ≈ 0.88
```

### Check Long-Memory Norm

```python
long_out = model.delta_mem(x)
print(f"long_out norm: {long_out.norm().item():.4f}")
```

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("goldfinch_ahn")
class GoldFinchAHNBlock(BaseAttention): ...
```

## B.8 Injection

Standard pattern with whole-model and single-layer traversal.

---
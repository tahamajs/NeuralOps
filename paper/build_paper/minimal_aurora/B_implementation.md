# Appendix B. Implementation Details

## B.1 The FactorisedLinear Module

```python
class FactorisedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=64,
        bias=True,
        energy=0.95,
        lora_rank=16,
        lora_alpha=32.0,
    ):
        super().__init__()
        self.rank = min(rank, in_features, out_features)
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))

        self.reset_parameters()

    def decompose_from_weight(self, W, bias=None, energy=None):
        U, S, Vt = torch.linalg.svd(W.float())
        target_energy = energy if energy is not None else self.energy
        cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
        calc_rank = int((cum_energy < target_energy).sum().item()) + 1
        r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
        r = max(1, r)

        sqrt_s = torch.sqrt(S[:r])
        A = U[:, :r] * sqrt_s
        B = sqrt_s[:, None] * Vt[:r, :]

        self.up.weight.data[:, :r] = A
        self.down.weight.data[:r, :] = B

        if bias is not None:
            self.up.bias.data[: len(bias)] = bias

        self.freeze_base_weights()
```

## B.2 Forward Pass

```python
def forward(self, x):
    out = self.up(self.down(x))
    if self.lora_rank > 0:
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
    return out
```

## B.3 Nested Encoder Output Projection

```python
elif proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

## B.4 The `pres_gate` (Unused)

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

Never referenced in `forward`. Recommend removal:

```python
# Remove this line
```

## B.5 The `output_scale`

```python
self.output_scale = nn.Parameter(torch.ones(1))
```

Applied in forward:

```python
output = self.out_proj(context) * self.output_scale
```

## B.6 Common Pitfalls

### B.6.1 Forgetting to Freeze Base

After `decompose_from_weight`, `freeze_base_weights` must be called.

### B.6.2 Nested Encoder Mis-Detection

DistilBERT's `MultiHeadSelfAttention` ends with "SelfAttention" but has 
`out_lin` → NOT nested. Detection must check `has_out_proj`.

### B.6.3 `output_scale` Drift

Initialize to 1.0; monitor during training.

### B.6.4 Silent Injection

`inject_minimal_aurora` returns `None` on failure.

---
# Appendix B. Implementation Details

## B.1 The ResSVDLinear Module

Located in `framework/compression_engine/res_svd.py`. The constructor 
signature:

```python
ResSVDLinear(
    in_features: int,
    out_features: int,
    base_rank: int,
    residual_rank: int,
    scaling: float = 1.0,
    factorize_base: bool = True,
    bias: bool = True,
)
```

The `from_linear` class method decomposes a given `nn.Linear`:

```python
@classmethod
def from_linear(cls, linear, base_rank, residual_rank, ...):
    W = linear.weight.data.float()
    
    # Base SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_b = U[:, :base_rank]
    S_b = S[:base_rank]
    Vt_b = Vt[:base_rank, :]
    
    # Residual
    W_b = (U_b * S_b) @ Vt_b
    R = W - W_b
    U_r, S_r, Vt_r = torch.linalg.svd(R, full_matrices=False)
    
    # LoRA init
    S_r_sqrt = torch.sqrt(S_r[:residual_rank])
    B_init = U_r[:, :residual_rank] * S_r_sqrt
    A_init = S_r_sqrt[:, None] * Vt_r[:residual_rank, :]
    
    layer = cls(...)
    layer.U_base.data = U_b
    layer.V_base.data = Vt_b
    layer.lora_B.data = B_init
    layer.lora_A.data = A_init
    if linear.bias is not None:
        layer.bias.data = linear.bias.data
    return layer
```

## B.2 Forward Pass

```python
def forward(self, x):
    # Base
    h_b = F.linear(x, self.V_base)
    y_b = F.linear(h_b, self.U_base)
    
    # Residual
    h_r = F.linear(x, self.lora_A)
    y_r = F.linear(h_r, self.lora_B)
    
    return y_b + self.scaling * y_r + self.bias
```

## B.3 Base Freezing

```python
self.U_base.requires_grad = False
self.V_base.requires_grad = False
```

Only `lora_A`, `lora_B`, and `bias` (optional) receive gradient.

## B.4 The `copy_weights` Procedure

For each of Q, K, V, O:

1. Extract the source weight.
2. Handle shape mismatch (transpose or repeat).
3. Create a temporary `nn.Linear`.
4. Call `ResSVDLinear.from_linear`.
5. Load state into the target `ResSVDLinear`.

**Verification.** After loading:

```python
sample_x = torch.eye(min(target.in_features, 64), target.in_features)
y = target(sample_x)
fnorm = torch.linalg.norm(y.float()).item()
if fnorm < 1e-4:
    raise RuntimeError(f"Weight copy failed for {proj_key}")
```

## B.5 Nested Encoder Special Case

For RoBERTa, the output projection is initialized to **identity**:

```python
if proj_key == "o" and self.is_nested_encoder:
    temp_linear = nn.Linear(target.in_features, target.out_features)
    nn.init.eye_(temp_linear.weight)
    nn.init.zeros_(temp_linear.bias)
    new_layer = ResSVDLinear.from_linear(temp_linear, ...)
    target.load_state_dict(new_layer.state_dict())
```

This makes the output projection a pass-through, so the outer 
`attention.output.dense` applies the actual projection.

## B.6 Common Pitfalls

### B.6.1 base_rank + residual_rank > min(d_in, d_out)

If base_rank + residual_rank exceeds the matrix rank, the residual SVD 
will have zero singular values beyond the original rank. This is fine 
but wasteful.

### B.6.2 Scaling Factor

The scaling factor $\alpha$ multiplies the residual. Default 1.0. Some 
setups prefer $\alpha = 1/r_r$ (like LoRA).

### B.6.3 Silent Injection Failure

The `inject_res_svd` hook returns `None` on failure. Always check the 
return value.

### B.6.4 Nested Encoder

RoBERTa requires identity init of output projection.

---
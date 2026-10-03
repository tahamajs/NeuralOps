# Appendix B. Implementation Details

## B.1 Core Module

```python
class HGRN2Layer(BaseAttention):
    def __init__(
        self,
        config,
        rank=None,
        d_model=None,
        res_clip=30.0,
        lora_rank=16,
        lora_alpha=32.0,
        is_decoder=None,
        freeze_base=True,
        **kwargs,
    ):
        super().__init__(config, rank)

        self.d_model = resolve_d_model(config, d_model, kwargs)
        self.res_clip = res_clip

        # Gates
        self.forget_gate = nn.Linear(self.d_model, self.d_model)
        self.input_gate = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.d_model)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # Init
        nn.init.normal_(self.forget_gate.weight, std=0.01)
        nn.init.constant_(self.forget_gate.bias, -2.5)
        nn.init.normal_(self.input_gate.weight, std=0.02)
        nn.init.constant_(self.input_gate.bias, 0.0)
        nn.init.normal_(self.g_proj.weight, std=0.02)
        nn.init.constant_(self.g_proj.bias, 1.28)
        nn.init.normal_(self.v_proj.weight, std=0.02)
```

## B.2 The Chunked Scan

```python
def _scan_direction(self, f, iv, chunk_size=32):
    B, L, D = iv.shape
    f_fp32 = f.float()
    iv_fp32 = iv.float()
    log_f = torch.log(f_fp32.clamp(min=1e-6))

    h_chunks = []
    h_prev = torch.zeros(B, D, device=iv.device, dtype=torch.float32)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        C = end - start

        f_c = log_f[:, start:end]  # (B, C, D)
        iv_c = iv_fp32[:, start:end]  # (B, C, D)

        A = torch.cumsum(f_c, dim=1)  # (B, C, D)
        A_diff = A.unsqueeze(2) - A.unsqueeze(1)  # (B, C, C, D)

        mask = torch.tril(torch.ones(C, C, device=iv.device, dtype=torch.bool))
        A_diff = torch.where(mask.unsqueeze(0).unsqueeze(-1), A_diff, torch.full_like(A_diff, -1e4))

        decay_weights = torch.exp(A_diff) * mask.unsqueeze(0).unsqueeze(-1)
        intra = torch.sum(decay_weights * iv_c.unsqueeze(1), dim=2)
        inter = h_prev.unsqueeze(1) * torch.exp(A)
        h_c = inter + intra

        if self.res_clip is not None and self.res_clip > 0:
            h_c = torch.clamp(h_c, -self.res_clip, self.res_clip)

        h_chunks.append(h_c)
        h_prev = h_c[:, -1]

    return torch.cat(h_chunks, dim=1).to(iv.dtype)
```

## B.3 Forget Gate Initialization

```python
nn.init.normal_(self.forget_gate.weight, mean=0.0, std=0.01)
nn.init.constant_(self.forget_gate.bias, -2.5)
```

**Why -2.5?** `softplus(-2.5) ≈ 0.076`. So `log f = -0.076`, 
`f = e^{-0.076} ≈ 0.93`. This means the state retains 93% of its value 
per step — slow decay.

## B.4 Input Modulation

```python
input_scale = (1.0 - f) * i
iv = input_scale * v
```

**Why (1-f) * i?** When $f \approx 1$ (retain), $1 - f \approx 0$, so no 
input. When $f \approx 0$ (forget), $1 - f \approx 1$, so full input. This 
ties the write magnitude to the forget rate.

## B.5 Output Modulation

```python
h_norm = self.state_norm(h_ctx)
gated = (0.5 * v + 0.5 * h_norm) * F.silu(z)
out = self.out_proj(gated)
```

**Why 0.5 v + 0.5 h_norm?** Preserves the current value alongside the 
recurrent state.

## B.6 Failure Modes

### B.6.1 Missing V/O

If V/O weights cannot be extracted, `copy_weights` logs a warning but does 
not raise. This is a known limitation.

### B.6.2 Nested Encoder

For nested encoders, we set `layer.attention.output.dense = nn.Identity()`. 
If this is wrong, the output projection is skipped.

### B.6.3 Cross-Attention

Cross-attention raises `NotImplementedError`. This is intentional.

## B.7 Common Pitfalls

### B.7.1 Random Gate Initialization

Without careful init, forget gates can saturate.

### B.7.2 FP16 Scan

FP16 scan loses precision. We use FP32 for the scan.

### B.7.3 Missing Clamp

Without `res_clip`, states can grow unbounded.

### B.7.4 Wrong Nested Handling

For RoBERTa, forgetting to set `output.dense = nn.Identity()` causes 
double projection.

---
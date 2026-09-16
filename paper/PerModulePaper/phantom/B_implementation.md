<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code-level details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `FactorisedLinear_CUR` Module

```python
class FactorisedLinear_CUR(nn.Module):
    def __init__(self, in_features, out_features, cur_rank=16, ...):
        super().__init__()
        self.down = nn.Linear(in_features, cur_rank, bias=False)
        self.up = nn.Linear(cur_rank, out_features, bias=True)
        # LoRA (optional)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## B.2 The `copy_cur` Method

```python
def copy_cur(self, weight, bias):
    with torch.no_grad():
        W = weight.float()
        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Leverage
        col_lev = (Vt ** 2).sum(dim=0)
        row_lev = (U ** 2).sum(dim=1)
        # Top-r selection
        rank = min(self.cur_rank, W.shape[0], W.shape[1])
        col_idx = torch.topk(col_lev, k=rank).indices
        row_idx = torch.topk(row_lev, k=rank).indices
        # CUR factors
        C = W[:, col_idx]
        R = W[row_idx, :]
        W_int = W[row_idx][:, col_idx]
        U_int_pinv = torch.linalg.pinv(W_int)
        # Assign
        self.down.weight.data.zero_()
        self.down.weight.data[:rank].copy_(R)
        self.up.weight.data.zero_()
        self.up.weight.data[:, :rank].copy_(C @ U_int_pinv)
        if bias is not None:
            self.up.bias.data.copy_(bias)
    if self.freeze_base:
        self.freeze_base_weights()
```

## B.3 LSH Attention

```python
def lsh_attention(Q, K, V, rounds, bucket_size, softpick_temp, ...):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    n_bits = max(1, min(bucket_size, 8))
    bit_shifts = torch.tensor([2**i for i in range(n_bits)], device=Q.device)
    
    Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
    K_norm = F.normalize(K, p=2, dim=-1, eps=1e-5)
    
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    # mask padding
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e4)
    # causal
    if is_causal:
        causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), -1e4)
    
    accum = torch.zeros_like(V)
    for _ in range(rounds):
        proj = F.normalize(torch.randn(D, n_bits, device=Q.device, dtype=Q.dtype), p=2, dim=0)
        hq = (torch.matmul(Q_norm, proj) > 0).float() @ bit_shifts
        hk = (torch.matmul(K_norm, proj) > 0).float() @ bit_shifts
        match = (hq.unsqueeze(-1) == hk.unsqueeze(-2))
        eye = torch.eye(N, device=Q.device, dtype=torch.bool)
        match = match | eye
        if is_causal:
            match = (match & ~causal.unsqueeze(0).unsqueeze(0)) | eye
        scores_r = scores.masked_fill(~match, -1e4)
        attn = softpick(scores_r, dim=-1, temperature=softpick_temp).to(Q.dtype)
        if dropout is not None:
            attn = dropout(attn)
        accum += torch.matmul(attn, V)
    
    return accum / rounds
```

## B.4 The `softpick` Function

```python
def softpick(x, dim=-1, temperature=1.0):
    orig = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    temp = max(temperature, 1e-4)
    e = torch.exp((x - m) / temp)
    m_scaled = torch.clamp(-m / temp, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    p_sum = p.sum(dim=dim, keepdim=True)
    fallback = F.softmax(x / temp, dim=dim)
    res = torch.where(p_sum > 1e-5, p / torch.clamp(p_sum, min=1e-5), fallback)
    return res.to(orig)
```

## B.5 Handling Conv1D (GPT-2)

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    c_w = c_w.t()  # (in, out) → (out, in)
```

## B.6 Handling Nested Encoder (RoBERTa)

```python
if hasattr(src, "self") and hasattr(src, "output"):
    self.is_nested_encoder = True
    # Extract from src.self.query, key, value
    # Do NOT extract output.dense
```

## B.7 Handling GQA

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    K = repeat_kv(K, repeats)
    V = repeat_kv(V, repeats)
```

## B.8 Common Pitfalls

### B.8.1 Missing L2 Normalization

LSH requires normalized vectors for angular hashing:

```python
Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
```

### B.8.2 Wrong Bit Shifts

Bits must be weighted by powers of 2:

```python
bit_shifts = torch.tensor([2**i for i in range(n_bits)])
```

### B.8.3 No Diagonal Mask

Without the eye matrix, a token may not attend to itself:

```python
match = match | eye
```

### B.8.4 Wrong Causal Mask Direction

Causal mask for decoder:

```python
causal = torch.triu(torch.ones(N, N), diagonal=1)
match = match & ~causal.unsqueeze(0).unsqueeze(0)
```

## B.9 Diagnostics

To verify injection:

```python
for name, mod in model.named_modules():
    if "attention" in name and "self" in name:
        print(name, type(mod).__name__)
        break
```

Expected output: `PHANTOMAttention`.

---
# Appendix B. Implementation Details

## B.1 Core Module

```python
class DecayLinearAttention(BaseAttention):
    def __init__(self, config, rank=None, d_model=None, num_heads=None, ...):
        super().__init__(config, rank)
        
        self.dim = ...
        self.num_heads = ...
        self.head_dim = ...
        self.num_kv_heads = ...
        
        # LoRA projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        # Data-dependent decay gate
        self.decay_gate = nn.Linear(self.dim, self.num_heads, bias=True)
        self.log_decay = nn.Parameter(
            torch.linspace(-0.5, 0.0, self.num_heads)
        )
        nn.init.zeros_(self.decay_gate.weight)
        nn.init.zeros_(self.decay_gate.bias)
```

## B.2 The Chunked Scan

```python
def _decay_scan_linear(self, q, k, v, lambdas, mask=None, reverse=False, chunk_size=32):
    B, H, L, d = q.shape
    
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    lambdas_f = lambdas.float().clamp(min=1e-4, max=1.0 - 1e-4)
    
    if mask is not None:
        m_f = mask.float().unsqueeze(1).unsqueeze(-1)
        k_f = k_f * m_f
        v_f = v_f * m_f
    
    if reverse:
        q_f = q_f.flip(dims=[2])
        k_f = k_f.flip(dims=[2])
        v_f = v_f.flip(dims=[2])
        lambdas_f = lambdas_f.flip(dims=[-1])
    
    log_lambdas = torch.log(lambdas_f)
    cum_log = torch.cumsum(log_lambdas, dim=-1)
    
    if L <= chunk_size:
        # Single chunk: dense
        ...
    else:
        # Multi-chunk: state-passing
        ...
```

## B.3 Intra-Chunk Computation

```python
# Reshape
qc = q_pad.view(B, H, num_chunks, chunk_size, d)
kc = k_pad.view(B, H, num_chunks, chunk_size, d)
vc = v_pad.view(B, H, num_chunks, chunk_size, d)
cum_c = cum_pad.view(B, H, num_chunks, chunk_size)

# Intra-chunk decay
intra_log_decay = cum_c.unsqueeze(-1) - cum_c.unsqueeze(-2)
intra_decay = torch.exp(intra_log_decay.clamp(max=0.0))
causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, ...))
intra_decay = intra_decay.masked_fill(~causal_mask, 0.0)

intra_scores = torch.matmul(qc, kc.transpose(-1, -2)) * intra_decay
num_intra = torch.matmul(intra_scores, vc)
denom_intra = intra_scores.sum(dim=-1, keepdim=True)
```

## B.4 Inter-Chunk Computation

```python
S = torch.zeros(B, H, d, d, ...)
z = torch.zeros(B, H, d, 1, ...)

for c in range(num_chunks):
    curr_cum = cum_c[:, :, c]
    
    if c == 0:
        inter_decay = torch.zeros(B, H, chunk_size, 1, ...)
    else:
        inter_decay = torch.exp((curr_cum - prev_cum_end).clamp(max=0.0)).unsqueeze(-1)
    
    num_inter = torch.matmul(qc[:, :, c], S) * inter_decay
    denom_inter = torch.matmul(qc[:, :, c], z) * inter_decay
    
    # Combine + update state
    ...
```

## B.5 Common Pitfalls

### B.5.1 Numeric Instability in Log

Clamp $\lambda$ to $(10^{-4}, 1 - 10^{-4})$ before log.

### B.5.2 Padding

Pad L to a multiple of chunk size, mask the padding in the last chunk.

### B.5.3 Mask Application

Apply attention mask to K, V **before** the scan, not after.

### B.5.4 Reverse Scan

Use `.flip(dims=[2])` before and after; the mask must also be flipped.

### B.5.5 Cross-Attention

Do not apply bidirectional scan for cross-attention.

## B.6 Memory Footprint

State variables per layer:
- $S \in \mathbb{R}^{B \times H \times d_h \times d_h}$: $8 \times 12 \times 64 \times 64 = 393,216$ floats = **1.5 MB** at FP32.
- $z \in \mathbb{R}^{B \times H \times d_h \times 1}$: negligible.

**Total per layer:** ~1.5 MB. Independent of $L$.

## B.7 Optimization Notes

- We use `torch.cumsum` for the intra-chunk cumsum.
- The inter-chunk loop has `num_chunks = L / C` iterations. For $L = 128$, $C = 32$: **4 iterations**.
- We can parallelize across the batch, head, and chunk dimensions.

## B.8 Feature Map Compatibility

The ELU+1 feature map is applied to Q, K after projection:

```python
q = F.elu(q) + 1.0
k = F.elu(k) + 1.0
```

Applied **before** the chunked scan.

---
# Appendix B. Implementation Details

## B.1 Softpick

```python
def softpick(x, dim=-1, temperature=1.0):
    orig_dtype = x.dtype
    temp = max(temperature, 1e-6)
    x_scaled = torch.clamp(x.float() / temp, max=85.0)
    p = F.relu(torch.exp(x_scaled) - 1.0)
    denom = p.sum(dim=dim, keepdim=True) + 1e-8
    return (p / denom).to(orig_dtype)
```

## B.2 XAttention Scoring

```python
@staticmethod
def score_blocks(Q, K, block_size, stride=2):
    B, H, N, D = Q.shape
    n_blocks = (N + block_size - 1) // block_size
    Q_sample = Q[:, :, ::stride, :]
    K_sample = K[:, :, ::stride, :]
    attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * (D**-0.5)
    scores = torch.zeros(B, H, n_blocks, n_blocks)
    # ... block-wise mean
    return scores
```

## B.3 Seer Gating

```python
class SeerGating(nn.Module):
    def __init__(self, head_dim, num_heads, gate_rank, top_k, temperature):
        self.q_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )
        self.k_pool = nn.Sequential(...)
    
    def forward(self, Q, K, n_blocks, training):
        Q_pooled = Q.mean(dim=2)
        K_pooled = K.mean(dim=2)
        q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)
        k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)
        gate_score = torch.sigmoid((q_gate @ k_gate).squeeze() / T)
        keep_per_head = torch.clamp((gate_score * n_blocks).long(), 1, n_blocks)
        # Build mask
        ...
```

## B.4 Common Pitfalls

### B.4.1 Softpick Collapse

For all-negative logits, softpick produces uniform. Ensure masks are applied 
before softpick to avoid this.

### B.4.2 Stripe Inactive

Stripe requires $N > 256$. At $N = 128$ (SST-2), it's inactive.

### B.4.3 RFF Inactive

RFF requires `enable_spectraformer=True` with pretrained weights.

### B.4.4 Dense Fallback for Causal

Causal path always uses dense. No sparsity for GPT-2.

### B.4.5 CLS Block Preservation

Block 0 always retained via `inter_mask[:, :, :, 0] = True`.

---
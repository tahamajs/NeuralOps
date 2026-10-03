# Appendix B. Implementation Details

## B.1 STE Quantization

```python
def ste_quantize_4bit(x):
    if x.numel() == 0:
        return x
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    scale = torch.clamp((x_max - x_min) / 15.0, min=1e-5)
    x_q_int = torch.clamp(torch.round((x - x_min) / scale), 0, 15)
    x_q = x_q_int * scale + x_min
    return x + (x_q - x).detach()  # STE
```

## B.2 Delta-KV

```python
def simulate_paged_delta_kv(K, V, rank):
    B, H, N, d = K.shape
    if N <= 1:
        return K, V
    
    K_hat = torch.zeros_like(K)
    V_hat = torch.zeros_like(V)
    K_hat[:, :, 0] = K[:, :, 0]
    V_hat[:, :, 0] = V[:, :, 0]
    
    delta_K = K[:, :, 1:] - K[:, :, :-1]
    delta_V = V[:, :, 1:] - V[:, :, :-1]
    
    delta_K_hat = low_rank_svd_quantize(delta_K, rank)
    delta_V_hat = low_rank_svd_quantize(delta_V, rank)
    
    K_hat[:, :, 1:] = K[:, :, 0:1] + torch.cumsum(delta_K_hat, dim=2)
    V_hat[:, :, 1:] = V[:, :, 0:1] + torch.cumsum(delta_V_hat, dim=2)
    
    return K_hat, V_hat
```

## B.3 Router

```python
# Page means
q_bar = (q_blocks * valid_b.unsqueeze(-1)).sum(dim=3) / valid_b.sum(dim=3, keepdim=True).clamp(
    min=1
)
k_bar = (k_blocks * valid_b.unsqueeze(-1)).sum(dim=3) / valid_b.sum(dim=3, keepdim=True).clamp(
    min=1
)

# Router logits
router_logits = torch.matmul(self.W_rg(q_bar), k_bar.transpose(-1, -2)) / math.sqrt(d)

# CLS boost
router_logits_biased = router_logits.clone()
router_logits_biased[:, :, :, 0] += 1e4

# Top-K
_, topk_indices = torch.topk(router_logits_biased, k_experts, dim=-1)
topk_logits = torch.gather(router_logits, -1, topk_indices)
g_j = F.softmax(topk_logits, dim=-1)
```

## B.4 Common Pitfalls

### B.4.1 STE Only Affects Backward

Forward uses quantized; backward uses identity. If you inspect the forward 
values, they will be quantized, but gradients flow through the original.

### B.4.2 Delta-KV Requires Same Shape

Reconstructed $\hat{K}$ has shape $[B, H, N, d_h]$, same as $K$. No memory 
savings.

### B.4.3 6D Gather is Expensive

The intermediate `[B, H, N_p, k_exp, P, d_h]` can be 256 MB per layer.

### B.4.4 Page Padding

When $N$ is not divisible by $P$, pad with zeros. Mask padding keys.

### B.4.5 CLS Boost Must Exceed All Others

Use $10^4$, not $10^2$. Small boosts may not guarantee top-k.

---
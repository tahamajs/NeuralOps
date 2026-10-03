<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `PrimalAttention` Class

```python
class PrimalAttention(BaseAttention):
    def __init__(self, config=None, rank=64, pinv_iters=6, pretrained_attn=None, **kwargs):
        super().__init__(config, rank)
        self.dim = getattr(config, "hidden_size", 768)
        self.num_heads = getattr(config, "num_attention_heads", 12)
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.dim // self.num_heads
        self.rank = getattr(comp_config, "rank", rank)
        self.pinv_iters = pinv_iters
        
        self.query = LoRALinear(self.dim, self.all_head_size, ...)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, ...)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, ...)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, ...)
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Landmark Pooling

```python
def pool_to_landmarks(x, m):
    """Segment pooling: divide into m chunks, average each."""
    B, H, L, d = x.shape
    if L % m == 0:
        s = L // m
        return x.reshape(B, H, m, s, d).mean(dim=3)
    else:
        idx = torch.linspace(0, L - 1, m, dtype=torch.long, device=x.device)
        return x[:, :, idx, :]
```

## B.3 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, L, D = hidden_states.shape
    
    # 1. Project
    Q = self.query(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
    K = self.key(hidden_states).view(B, L, G, d).permute(0, 2, 1, 3)
    V = self.value(hidden_states).view(B, L, G, d).permute(0, 2, 1, 3)
    
    # 2. RoPE
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
    # 3. GQA repeat
    if H != G:
        K = repeat_kv(K, H // G)
        V = repeat_kv(V, H // G)
    
    # 4. Landmark pooling
    m = min(self.rank, L)
    Q_land = pool_to_landmarks(Q, m)
    K_land = pool_to_landmarks(K, m)
    
    # 5. Compute factors
    scale = 1.0 / (d ** 0.5)
    F1 = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
    F2 = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
    F3 = torch.matmul(Q_land, K.transpose(-2, -1)) * scale
    
    # 6. Pseudoinverse
    F2_inv = torch.linalg.pinv(F2.float(), rcond=1e-4)
    I = torch.eye(m, device=Q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    for _ in range(min(self.pinv_iters, 4)):
        F2_inv = torch.matmul(F2_inv, 2 * I - torch.matmul(F2.float(), F2_inv))
    
    # 7. Approximate attention
    attn = torch.matmul(F1, torch.matmul(F2_inv, F3))
    
    # 8. Causal mask
    if self.is_decoder:
        causal = torch.triu(torch.full((L, L), -1e4), diagonal=1)
        attn = attn + causal.unsqueeze(0).unsqueeze(0)
    
    # 9. Attention mask
    if attention_mask is not None:
        attn = apply_attention_mask(attn, attention_mask)
    
    # 10. Softmax
    probs = F.softmax(attn, dim=-1).to(hidden_states.dtype)
    probs = torch.nan_to_num(probs)
    probs = self.dropout(probs)
    
    # 11. Context
    context = torch.matmul(probs, V)
    context = context.permute(0, 2, 1, 3).contiguous().view(B, L, H * d)
    
    if self.is_nested_encoder:
        return context, None
    
    return self.out_proj(context), None
```

## B.4 `copy_weights`

Handles four architectures: GPT-2 (fused Conv1D), RoBERTa (nested), DistilBERT 
(flat), Qwen (GQA + RoPE). See §5.2.

## B.5 Common Pitfalls

### B.5.1 FP16 Pseudoinverse

`torch.linalg.pinv` requires FP32/FP64. Cast to FP32, compute, cast back.

### B.5.2 Schulz Divergence

Starting from `X_0 = 0` often diverges. Start from `X_0 = pinv(F2)`.

### B.5.3 RoPE After Pooling

Apply RoPE **before** pooling. Otherwise, pooled positions lose their 
absolute position information.

### B.5.4 GQA Repeat Before Pooling

Repeat KV heads **before** pooling. Otherwise, pooled landmarks don't match 
query head count.

### B.5.5 Causal Mask After Approximation

Apply causal mask to `attn` (the approximate matrix), not to `F1, F2, F3`. 
This is essential for the linear complexity.

---

# Appendix B. Implementation Details

## B.1 Core Forward

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, N, _ = hidden_states.shape
    
    q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, N, H_kv, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, N, H_kv, d).transpose(1, 2)
    
    if position_embeddings is not None:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    if self.num_heads != self.num_kv_heads:
        k = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_heads // self.num_kv_heads)
    
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    invalid_mask = torch.zeros(B, 1, N, N, dtype=torch.bool)
    if attention_mask is not None:
        valid_tok = get_token_valid_mask(attention_mask, B, N)
        invalid_mask = invalid_mask | (~valid_tok.unsqueeze(1).unsqueeze(2))
    
    if self.is_decoder:
        causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
        invalid_mask = invalid_mask | (~causal.unsqueeze(0).unsqueeze(0))
    
    fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
    masked = scores.masked_fill(invalid_mask, fill_val)
    max_scores = masked.max(dim=-1, keepdim=True).values
    
    thresh = -5e3 if scores.dtype in (torch.float16, torch.bfloat16) else -1e8
    max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
    
    centered = (scores - max_scores) / max(self.temperature, 1e-6)
    base = torch.clamp(1.0 + centered / self.order, min=0.0)
    attn = base ** self.order
    attn = attn.masked_fill(invalid_mask, 0.0)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    
    if head_mask is not None:
        attn = attn * head_mask
    
    attn = self.dropout(attn)
    context = torch.matmul(attn, v)
    context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, None
    return self.out_proj(context), None
```

## B.2 Common Pitfalls

### B.2.1 No Centering → Overflow

Without max-centering, base can exceed FP16 max.

### B.2.2 No ReLU → NaN

Negative base to fractional power gives NaN.

### B.2.3 All-Masked → NaN

If a query has no valid keys, max = -1e4, base = 0, sum = 0 → NaN. 
Guard with `max = 0`.

### B.2.4 Wrong Mask Order

Mask for max BEFORE polynomial; zero mask AFTER polynomial.

### B.2.5 Fragile RoPE Check

The `'q_rot' in locals()` check is fragile. Should compute `q_rot, q_pass` 
unconditionally.

---
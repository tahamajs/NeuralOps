<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `AdaptiveSparseAttention` Class

```python
class AdaptiveSparseAttention(BaseAttention):
    def __init__(self, config=None, rank=None, sparsity_ratio=None, ...):
        super().__init__(config, rank)
        
        # ... resolve hidden_size, num_heads, head_dim ...
        
        # Per-head sparsity logits
        init_logit = logit(effective_sparsity)
        self.sparsity_logits = nn.Parameter(
            torch.full((self.num_heads,), fill_value=init_logit)
        )
        
        # Projections with LoRA
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 Soft Top-$k$ Implementation

```python
# Inside forward, per chunk:
sorted_scores, _ = torch.sort(scores_chunk, dim=-1, descending=True)
k_indices = k_indices_base.expand(B, H, C_q, 1)
topk_thresh = torch.gather(sorted_scores, dim=-1, index=k_indices)

# Soft sigmoid mask
temperature = 0.1
soft_mask = torch.sigmoid((scores_chunk - topk_thresh) / temperature)
scores_chunk = scores_chunk + torch.log(soft_mask + 1e-8)
```

## B.3 Chunked Loop

```python
context_chunks = []
for q_start in range(0, N_q, chunk_size):
    q_end = min(q_start + chunk_size, N_q)
    q_chunk = Q[:, :, q_start:q_end, :]
    
    scores_chunk = torch.matmul(q_chunk, K.transpose(-2, -1)) * scale
    
    # ... mask, top-k, softmax ...
    
    ctx_chunk = torch.matmul(attn_chunk, V)
    context_chunks.append(ctx_chunk)

context = torch.cat(context_chunks, dim=2)
```

## B.4 Per-Head Budget Computation

```python
sparsity_per_head = torch.sigmoid(self.sparsity_logits).clamp(0.0, 0.95)
k_per_head = torch.clamp(
    ((1.0 - sparsity_per_head) * N_k).long(),
    min=1, max=N_k
)
k_indices_base = (k_per_head - 1).view(1, H, 1, 1)
```

## B.5 Handling T5 Relative Position Bias

```python
position_bias = kwargs.get("position_bias", None)
if position_bias is None and self.has_relative_attention_bias:
    position_bias = self.compute_bias(N_q, N_k, device=device)

# In the chunk loop:
if position_bias is not None:
    pb_chunk = position_bias[:, :, q_start:q_end, :]
    scores_chunk = scores_chunk + pb_chunk
```

## B.6 Common Pitfalls

### B.6.1 Missing FP32 Cast

`torch.log(soft_mask + 1e-8)` must be FP32 for stability.

### B.6.2 Wrong Causal Masking Position

Causal mask must be applied **per chunk**, not globally.

### B.6.3 Forgetting RoPE Before Chunking

RoPE must be applied to Q, K **before** the chunk loop.

### B.6.4 Zero-Size Threshold

If $k_h = 0$, the threshold indexing fails. We clamp $k_h \ge 1$.

### B.6.5 Numerically Unstable Log

`log(sigmoid(x))` is stable; `log(sigmoid(x) + 1e-8)` adds a small offset.

---
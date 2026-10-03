<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `NystromAttention` Class

```python
class NystromAttention(BaseAttention):
    def __init__(self, config=None, rank=None, num_landmarks=None,
                 pinv_iterations=3, pretrained_attn=None, old_attn=None,
                 comp_config=None, **kwargs):
        # Dimensions
        self.dim = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = ...
        self.all_head_size = ...
        self.all_kv_head_size = ...
        self.kv_repeat = ...
        
        # Landmarks
        self.num_landmarks = min(32, int(num_landmarks or 32))
        self.pinv_iterations = pinv_iterations
        
        # Decoder detection
        self.is_decoder = ...
        self.is_nested_encoder = False
        
        # LoRA config
        self.lora_rank = ...
        
        # Projections
        self.query = LoRALinear(...)
        self.key = LoRALinear(...)
        self.value = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 The Iterative Pseudoinverse

```python
def _iterative_pinv(self, A, iters=3):
    try:
        return torch.linalg.pinv(A.float()).to(A.dtype)
    except Exception:
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
        
        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        
        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V
```

**Note.** We first try `torch.linalg.pinv` (SVD). If it fails, we fall back 
to Newton-Schulz. This is a deviation from the original paper.

## B.3 The Three Execution Paths

```python
def forward(self, hidden_states, attention_mask=None, ...):
    Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
    K = self.key(hidden_states).view(B, N, Hkv, d).transpose(1, 2)
    V = self.value(hidden_states).view(B, N, Hkv, d).transpose(1, 2)
    
    # RoPE
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
    # GQA
    if self.kv_repeat > 1:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    valid_mask = get_token_valid_mask(attention_mask, B, N, device)
    
    # PATH 1: Causal decoder
    if self.is_decoder or self.is_causal:
        # Dense causal
        ...
        return out, None
    
    # PATH 2: Short sequence
    if N <= max(128, 2 * self.num_landmarks):
        context = self._dense_attention(Q, K, V, valid_mask, dtype)
        return out, None
    
    # PATH 3: Nyström
    # adaptive pooling landmarks
    Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m)...
    K_landmarks = F.adaptive_avg_pool1d(K_flat, m)...
    
    scores1 = Q @ K_landmarks^T
    scores2 = Q_landmarks @ K_landmarks^T
    scores3 = Q_landmarks @ K^T
    
    F1 = softmax(scores1); F2 = softmax(scores2); F3 = softmax(scores3)
    F2_pinv = self._iterative_pinv(F2)
    
    step1 = F3 @ V
    step2 = F2_pinv @ step1
    context = F1 @ step2
    
    # Head mask, valid mask, out_proj
    ...
```

## B.4 Landmark Selection via Adaptive Pooling

```python
m = min(self.num_landmarks, N)

Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)   # (BH, d, N)
K_flat = K.reshape(B * H, N, d).transpose(1, 2)

Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
K_landmarks = F.adaptive_avg_pool1d(K_flat, m).transpose(1, 2).reshape(B, H, m, d)
```

**Why adaptive_avg_pool1d?** It partitions the sequence into $m$ segments 
of approximately equal length and averages each. This is deterministic and 
preserves locality.

## B.5 Weight Transfer

Four patterns:

1. **GPT-2** — split fused `c_attn`, `c_proj` for output
2. **RoBERTa** — `self.query/key/value`, `output.dense`
3. **Qwen/DistilBERT** — flat projections
4. **UniversalAdapter** — fallback

## B.6 Injection Hook

```python
def inject_nystrom(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or layer.attention...
    hidden_size = ...
    num_heads = ...
    
    try:
        landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config else 32
        new_attn = NystromAttention(
            config=model_config,
            num_landmarks=landmarks,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"injection failed: {e}")
        return None
```

## B.7 Common Pitfalls

### B.7.1 Causal Without Fallback

Using Nyström for causal models leaks future information. Must use 
dense causal.

### B.7.2 No Short Sequence Fallback

For $N \leq 2m$, dense is faster and more accurate.

### B.7.3 Conv1D Not Transposed

GPT-2's Conv1D stores `(in, out)`; must transpose.

### B.7.4 Pinv Without Preconditioning

Newton-Schulz requires $\|\alpha A\| < 1$. Without preconditioning, it 
diverges.

### B.7.5 Wrong Associative Order

Computing $F_1 \cdot \text{pinv}(F_2)$ before $F_3 V$ gives $\mathcal{O}(N^2 m)$, 
defeating the purpose.

## B.8 Numerical Stability

- Pinv computed in FP32.
- `torch.nan_to_num` guards.
- Landmark pooling in input dtype.

## B.9 Comparison with Companion Methods

| Feature | Dobi | SVD-LLM V2 | Nyström |
|---|---|---|---|
| Reduces | Params | Params | **Compute** |
| Landmarks | ❌ | ❌ | ✅ |
| Iterative pinv | ❌ | ❌ | ✅ |
| Causal | ✅ | ✅ | **❌** |
| Short seq | ✅ | ✅ | **Fallback** |
| Code lines | ~450 | ~600 | ~450 |

---
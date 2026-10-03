<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `GroupedQueryAttention` Class

```python
class GroupedQueryAttention(BaseAttention):
    def __init__(
        self,
        config=None,
        rank=None,
        dim=None,
        num_heads=None,
        num_kv_heads=None,
        lora_rank=16,
        lora_alpha=32.0,
        pretrained_attn=None,
        old_attn=None,
        **kwargs,
    ):
        super().__init__(config, rank)
        src_attn = pretrained_attn or old_attn

        # Resolve dim, num_heads, num_kv_heads
        self.hidden_size = ...
        self.num_heads = ...
        self.num_kv_heads = min(self.num_heads, max(1, num_kv_heads))

        self.head_dim = self.hidden_size // self.num_heads
        self.kv_repeat = self.num_heads // self.num_kv_heads

        # Projections
        self.query = LoRALinear(self.hidden_size, self.all_head_size, ...)
        self.key = LoRALinear(self.hidden_size, self.kv_head_size, ...)
        self.value = LoRALinear(self.hidden_size, self.kv_head_size, ...)
        self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, ...)

        if src_attn is not None:
            self.copy_weights(src_attn)
```

## B.2 `copy_weights`

```python
def copy_weights(self, src):
    extracted = {"q": (None, None), "k": None, "v": None, "o": None}
    
    # Extract from various architectures
    if hasattr(src, "c_attn"):  # GPT-2
        w, b = _unpack_weight_bias(src.c_attn)
        w = w.t()  # Conv1D transpose
        q_w, k_w, v_w = w.chunk(3, dim=0)
        ...
    elif hasattr(src, "self"):  # RoBERTa nested
        q_w, b_q = _unpack_weight_bias(src.self.query)
        ...
    elif hasattr(src, "q_lin"):  # DistilBERT
        q_w, b_q = _unpack_weight_bias(src.q_lin)
        ...
    
    # Handle MHA → GQA conversion
    # Query: direct copy
    # K, V: group-average
    
    # Copy into projections
    with torch.no_grad():
        for proj_key, target in [("q", self.query), ("k", self.key), ...]:
            w, b = extracted[proj_key]
            if w is not None:
                # Shape-aware copy
                ...
                target.freeze_base_weights()
```

## B.3 Group Averaging

```python
def group_average(w_mha, H_src, G_tgt, d_h):
    """Average KV heads into G groups."""
    r = H_src // G_tgt
    w_grouped = w_mha.view(d_h * G_tgt, r, ...).mean(dim=1)
    return w_grouped
```

## B.4 `repeat_kv`

```python
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, G, L, d_h = x.shape
    x = x[:, :, None, :, :].expand(B, G, n_rep, L, d_h)
    return x.reshape(B, G * n_rep, L, d_h)
```

## B.5 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, N_q, D = hidden_states.shape
    
    Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2)
    K = self.key(hidden_states).view(B, N_q, G, d).transpose(1, 2)
    V = self.value(hidden_states).view(B, N_q, G, d).transpose(1, 2)
    
    if RoPE:
        Q, K = apply_rotary_pos_emb(Q, K, ...)
    
    if self.kv_repeat > 1:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    scores = Q @ K.transpose(-2, -1) / sqrt(d)
    
    if causal:
        scores = scores.masked_fill(~causal_tril, -10000.0)
    
    attn = softmax(scores)
    context = attn @ V
    context = context.transpose(1, 2).view(B, N_q, H * d)
    
    if self.is_nested_encoder:
        return context, None
    
    output = self.out_proj(context)
    return output, None
```

## B.6 Common Pitfalls

### B.6.1 Wrong Transpose for Conv1D

```python
# WRONG
w = src.c_attn.weight  # (in, out) — no transpose

# RIGHT
w = src.c_attn.weight.t()  # (out, in)
```

### B.6.2 Missing Divide Check

$G$ must divide $H$. If not, snap to largest divisor.

### B.6.3 Causal Mask Before Repetition

Same result, but apply after for clarity.

### B.6.4 RoPE After Repetition

RoPE must be applied **before** repetition (to shared KV), not after.

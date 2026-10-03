<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `SqueezingHeadsDistiller` Class

```python
class SqueezingHeadsDistiller(BaseAttention):
    def __init__(self, config=None, old_attn=None, rank=None,
                 lora_rank=16, lora_alpha=32.0, is_decoder=None, **kwargs):
        super().__init__(config, rank)
        
        # Resolve dims
        self.hidden_size = ...
        self.num_heads = ...
        self.head_dim = ...
        self.num_kv_heads = ...
        
        # LoRA-wrapped projections
        self.q_proj = LoRALinear(self.hidden_size, self.q_head_size,
                                  lora_rank=lora_rank, lora_alpha=lora_alpha,
                                  freeze_base=True)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if old_attn is not None:
            self.copy_weights(old_attn)
```

## B.2 copy_weights — Core Logic

```python
def copy_weights(self, old_attn):
    # Detect nested encoder
    has_out_lin = hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj")
    if has_out_lin:
        self.is_nested_encoder = False
    elif hasattr(old_attn, "self") and hasattr(old_attn, "output"):
        self.is_nested_encoder = True
    
    # Extract Q, K, V, O per architecture
    if hasattr(src, "c_attn"):  # GPT-2
        w = src.c_attn.weight.t()
        q_w, k_w, v_w = w.chunk(3, dim=0)
    elif hasattr(src, "query"):  # RoBERTa
        q_w = src.query.weight
        k_w = src.key.weight
        v_w = src.value.weight
    elif hasattr(src, "q_proj"):  # Qwen
        q_w = src.q_proj.weight
        ...
    elif hasattr(src, "q_lin"):  # DistilBERT
        q_w = src.q_lin.weight
        ...
    
    # Copy with head-dim slicing
    self.q_proj.weight.zero_()
    m_out = min(self.q_head_size, w_q.shape[0])
    m_in = min(self.hidden_size, w_q.shape[1])
    self.q_proj.weight[:m_out, :m_in].copy_(w_q[:m_out, :m_in])
    
    # Copy with GQA pooling for K, V
    if w_k.shape[0] > self.kv_head_size:
        rep = w_k.shape[0] // self.kv_head_size
        w_k_pooled = w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1)
        self.k_proj.weight.copy_(w_k_pooled)
```

## B.3 Common Pitfalls

### B.3.1 Wrong `is_nested_encoder`

If RoBERTa's `is_nested_encoder` is False, `out_proj` is applied inside 
AND outside, doubling the output.

```python
# Correct check:
if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj"):
    self.is_nested_encoder = False
elif hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
```

### B.3.2 Wrong Conv1D Transpose

GPT-2 Conv1D stores weights as `(in, out)`. Must transpose:

```python
c_attn_w = src.c_attn.weight.detach()
if "Conv1D" in cls_name or hasattr(src.c_attn, "nf"):
    c_attn_w = c_attn_w.t()
```

### B.3.3 GQA Pooling Shape

When pooling GQA heads, the reshape must be exact:

```python
w_k_pooled = (
    w_k[:, :self.hidden_size]
    .view(self.num_kv_heads, rep, self.head_dim, -1)
    .mean(dim=1)
    .view(self.kv_head_size, -1)
)
```

### B.3.4 Missing Frobenius Check

Without checking weight norms, silently failing copies go unnoticed.

### B.3.5 LoRA Not Frozen Correctly

Ensure `weight.requires_grad = False` but `lora_A/B.requires_grad = True`.

---

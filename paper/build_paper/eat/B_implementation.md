<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The EATBlock Module

```python
@ATTENTION_REGISTRY.register("eat")
@ATTENTION_REGISTRY.register("eat_attention")
class EATBlock(BaseAttention):
    def __init__(self, config, rank=64, ..., prune_layer=None,
                 pretrained_attn=None, ...):
        super().__init__(config, rank)
        self.q_lin = LoRALinear(d, H*d_h, lora_rank=16)
        self.k_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.v_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_lin = LoRALinear(H*d_h, d, lora_rank=16)
        self.prune_layer = prune_layer
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
    
    def forward(self, X, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, position_embeddings=None, **kwargs):
        B, L, D = X.shape
        # 1. Project
        q = self.q_lin(X).view(B, L, H, d_h).transpose(1, 2)
        k = self.k_lin(X).view(B, L, H_kv, d_h).transpose(1, 2)
        v = self.v_lin(X).view(B, L, H_kv, d_h).transpose(1, 2)
        # 2. RoPE
        if position_embeddings is not None:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        # 3. GQA
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)
        # 4. Attention
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if self.is_decoder:
            scores = scores + causal_mask
        if attention_mask is not None:
            scores = scores + pad_mask
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs.to(v.dtype), v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H*d_h)
        # 5. Output projection
        if self.is_nested_encoder:
            out = attn_out
        else:
            out = self.out_lin(attn_out)
        # 6. Optional pruning (inference only)
        if self.prune_layer is not None and not self.training:
            out, _ = self.prune_layer(out)
        return out, None
```

## B.2 The LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora
        return out
```

## B.3 Copy Weights — Case by Case

### B.3.1 DistilBERT

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### B.3.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

### B.3.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.data.t()  # transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### B.3.4 Qwen

```python
extracted["q"] = (src.q_proj.weight.data, src.q_proj.bias.data)
extracted["k"] = (src.k_proj.weight.data, src.k_proj.bias.data)
extracted["v"] = (src.v_proj.weight.data, src.v_proj.bias.data)
extracted["o"] = (src.o_proj.weight.data, src.o_proj.bias.data)
```

## B.4 The Silent Failure Path

**Critical code path** (in `copy_weights`):

```python
missing_after = [
    k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)
]
if missing_after:
    logger.warning(
        f"EATBlock copy_weights: missing projections {missing_after} "
        f"from {pretrained_attn.__class__.__name__}; "
        f"proceeding with random init for those"
    )
    # ← continues, doesn't raise
```

This is the origin of the "silent failure" problem. We recommend adding:

```python
if missing_after and os.environ.get("EAT_STRICT", "0") == "1":
    raise RuntimeError(f"Missing projections: {missing_after}")
```

## B.5 Post-Injection Sanity Check

```python
def sanity_check(model, device, seq_len=32):
    model.eval()
    ids = torch.randint(0, 1000, (2, seq_len), device=device)
    with torch.no_grad():
        out = model(ids).logits
    std = out.std(dim=0).mean().item()
    uniq = out.argmax(-1).unique().numel()
    if std < 1e-3 or uniq < 2:
        raise RuntimeError(f"Collapse detected: std={std}, uniq={uniq}")
```

Run this after every injection.

## B.6 Common Pitfalls

### B.6.1 Forgetting `is_nested_encoder`

For RoBERTa, must be `True`. Otherwise double projection.

### B.6.2 Missing Conv1D Transpose

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

### B.6.3 Not Freezing Base Weights

Base weights must have `requires_grad = False` after copy.

---
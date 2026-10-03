# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Cross-attn |
|---|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No | No |
| GPT-2 | `h[i].attn` | No | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes | No |
| T5 | `block[i].layer[0]` | No | No | No | Yes |

## 5.2 Weight Extraction Patterns

### 5.2.1 GPT-2 Conv1D

```python
c_w = c_attn.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat Modules

Direct attribute access on `q_lin` / `q_proj` / `query`.

### 5.2.4 UniversalAdapter Fallback

If any projection is missing.

## 5.3 Handling of the MoE Dimension

The MoE operates on `all_head_size = num_heads * head_dim`. This equals the 
hidden dimension for standard attention.

**For GQA models**, `kv_head_size < all_head_size`. But the attention output 
is always `all_head_size` (Q heads × head dim). So the MoE input dimension 
is always `all_head_size = d`.

## 5.4 Dynamic Dimension Adaptation

If the incoming hidden size differs from the router's input size, a 
**dynamic router** is created on-the-fly:

```python
if D == self.d_model:
    logits = self.router(self.router_ln(x_flat))
else:
    normed = F.layer_norm(x_flat, (D,))
    if not hasattr(self, "_dynamic_router") or self._dynamic_router.in_features != D:
        self._dynamic_router = nn.Linear(D, self.n_experts, bias=False)...
    logits = self._dynamic_router(normed)
```

This handles cross-attention where the input dimension may differ.

## 5.5 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`:

```python
if self.is_nested_encoder:
    if enriched_attn.shape[-1] != self.dim:
        out = self.out_lin(enriched_attn)
    else:
        out = enriched_attn
```

The outer `attention.output.dense` handles the projection.

## 5.6 Causality

For decoders, causal masking is applied to attention scores:

```python
if self.is_decoder and not is_cross_attention:
    causal_mask = torch.triu(torch.full((L, kv_len), -inf, ...), diagonal=1)
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
```

The MoE operates on the attention output, which is already causal.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject ExpertChoiceMoEAttention
```
Input:  model M, config c, n_experts, capacity_factor
Output: M with attention replaced

1: for each layer:
2:     src ← extract_attention(layer)
3:     if src is None: continue
4:     try:
5:         new ← ExpertChoiceMoEAttention(c, n_experts=E, ...)
6:         replace(layer, new)
7:     except Exception:
8:         raise RuntimeError("copy_weights failed")
9: return M
```

## 5.8 Failure Handling

The injection raises `RuntimeError` on any failure. This is strict because 
MoE injection is complex.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("expert_choice_moe")
@ATTENTION_REGISTRY.register("expert_choice_moe_attention")
class ExpertChoiceMoEAttention(BaseAttention):
    ...
```

---
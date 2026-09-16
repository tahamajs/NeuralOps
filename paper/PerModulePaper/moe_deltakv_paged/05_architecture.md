# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Bidirectional? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | Yes | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No |
| GPT-2 | `GPT2Attention` | No | No | No |

## 5.2 Weight Transfer

`copy_weights` extracts weights via five tiers:

### 5.2.1 Tier 1: UniversalAdapter

```python
for proj in ["q", "k", "v", "o"]:
    try:
        w, b = UniversalAdapter.get_weight_bias(pretrained_attn, proj)
        if w is not None:
            extracted[proj] = (w, b)
    except Exception:
        pass
```

### 5.2.2 Tier 2: GPT-2 Conv1D

```python
weight = pretrained_attn.c_attn.weight.data
if weight.shape[1] == 3 * hidden_size:
    weight = weight.t()
q_w, k_w, v_w = weight.chunk(3, dim=0)
```

### 5.2.3 Tier 3: RoBERTa Nested

```python
if hasattr(pretrained_attn, "self"):
    s = pretrained_attn.self
    extracted["q"] = (s.query.weight, ...)
    extracted["k"] = (s.key.weight, ...)
    extracted["v"] = (s.value.weight, ...)
```

### 5.2.4 Tier 4: Named Projections

```python
named_map = {
    "q": ["q_proj", "q_lin", "query"],
    "k": ["k_proj", "k_lin", "key"],
    "v": ["v_proj", "v_lin", "value"],
    "o": ["o_proj", "out_proj", "out_lin", "dense"],
}
```

### 5.2.5 Tier 5: Repetition for GQA

If source weight has smaller $d_\text{out}$ (GQA), we repeat:

```python
if w.shape[0] < self.all_head_size and self.all_head_size % w.shape[0] == 0:
    rep = self.all_head_size // w.shape[0]
    w = w.repeat(rep, 1)
```

## 5.3 Nested Encoder Detection

```python
is_nested = False
if hasattr(pretrained_attn, "self") and hasattr(getattr(pretrained_attn, "self"), "query"):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention") and not has_out_proj:
    is_nested = True
```

## 5.4 The Router $W_\text{rg}$

```python
self.W_rg = nn.Linear(self.head_dim, self.head_dim, bias=False)
nn.init.eye_(self.W_rg.weight)
```

Initialized to identity, so the router starts as the dot product of page-mean 
queries and keys.

## 5.5 Load Balancing Loss

Stored as a buffer (not a parameter):

```python
self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
```

Updated during forward:

```python
router_probs = F.softmax(router_logits, dim=-1)
expert_mask = F.one_hot(topk_indices, num_classes=num_pages).float()
density = expert_mask.sum(dim=-2).mean(dim=(0, 1, 2))
prob_mean = router_probs.mean(dim=(0, 1, 2))
self.aux_loss = float(num_pages) * torch.sum(density * prob_mean)
```

**Usage.** The training loop should add `aux_loss * λ` to the classification 
loss. Typical `λ = 0.01`.

## 5.6 The `is_causal` Flag

```python
self.is_causal = bool(
    getattr(config, "is_causal", False)
    or getattr(config, "is_decoder", False)
    or any(model_type.startswith(p) for p in ("gpt", "qwen", "llama", ...))
)
```

For causal models:
- Router logits are masked (upper-triangular).
- Full attention path is used when `num_pages <= 2`.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject MoE-DeltaKV-Paged
```
Input:  model M, config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     try:
4:         new ← MoEDeltaKVPagedAttention(config, pretrained_attn=src)
5:         replace(ℓ, new)
6:     except Exception as e:
7:         raise RuntimeError(...)  ▷ HARD FAIL
8: end for
```

**Note.** Current implementation raises on failure. In production, change 
to `logger.error(...); continue`.

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing` non-empty | **Raise** |
| Near-zero norm | Frobenius check | **Raise** |
| Injection fail | Exception | **Raise** |
| OOM | CUDA error | Abort |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("moe_deltakv_paged", supports_causal=True, supports_gqa=True)
class MoEDeltaKVPagedAttention(nn.Module):
    ...
```

## 5.10 Interaction with Recovery

Base weights are frozen; LoRA adapters and $W_\text{rg}$ are trainable.

Per-layer trainable:
- LoRA on query, key, value, out_proj: $4 r_L (d + d_h)$
- $W_\text{rg}$: $d_h^2$

For $d = 768$, $d_h = 64$, $r_L = 16$: ~102K per layer.

---
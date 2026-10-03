# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Transfer

We copy base weights from the pretrained attention via `copy_weights`, which 
handles four patterns:

### 5.2.1 Pattern 1: GPT-2 Conv1D

```python
c_w = src.c_attn.weight.t()  # Conv1D stores (in, out); transpose
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 Pattern 2: RoBERTa Nested

```python
if hasattr(src, "self") and hasattr(src, "output"):
    self.is_nested_encoder = True
    q_w = src.self.query.weight
    k_w = src.self.key.weight
    v_w = src.self.value.weight
    o_w = src.output.dense.weight
```

### 5.2.3 Pattern 3: Qwen/LLaMA (GQA)

```python
q_w = src.q_proj.weight  # (H·d_h, d)
k_w = src.k_proj.weight  # (H_kv·d_h, d)
v_w = src.v_proj.weight
o_w = src.o_proj.weight
```

### 5.2.4 Pattern 4: DistilBERT

```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### 5.2.5 Pattern 5: UniversalAdapter Fallback

If none of the above patterns match, we use `UniversalAdapter.get_weight_bias`.

## 5.3 Nested Encoder Bypass

For RoBERTa, we set `is_nested_encoder = True`, meaning our `forward` returns 
raw context and the outer `BertSelfOutput.dense` applies the output projection.

## 5.4 Causal Mask Construction

The most delicate part. For decoders:

```python
q_pos = torch.arange(L).view(1, 1, L, 1)
k_orig_pos = top_idx.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k)
causal_mask = (k_orig_pos > q_pos)
scores = scores.masked_fill(causal_mask, float("-inf"))
```

**Why this is correct.** Even though compressed keys are indexed $0..k-1$, 
their original positions `top_idx` are what matters. Query at position $i$ 
can attend only to keys whose original position is $\leq i$.

**Verification.** We check that output at position $i$ depends only on 
inputs $\leq i$ by computing gradients.

## 5.5 Handling Padding

Padding tokens are masked out of the score:

```python
total_importance = total_importance.masked_fill(mask == 0, -10000.0)
```

After selection, the compressed mask is gathered:

```python
comp_mask = torch.gather(mask, 1, top_indices)
```

## 5.6 Injection Algorithm

**Algorithm 2:** Inject TCA into a model
```
Input:  model M, compression_ratio ρ
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← TokenCompressedAttention(config, ρ, pretrained_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.7 The `score_net` Module

```python
self.score_net = nn.Sequential(
    nn.Linear(self.dim, 64),
    nn.GELU(),
    nn.Linear(64, 1),
)
```

Total parameters: `dim * 64 + 64 + 64 + 1 = 65*dim + 65`. For $d = 768$: 
**49,985 parameters** per layer.

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Weight extraction fails | Exception in copy_weights | Raise RuntimeError |
| Score net NaN | `torch.isnan` on scores | Fallback to norm-only |
| All tokens masked | `k_keep = 0` | Force k_keep = 1 |
| Short sequence | $L \leq 2$ | Skip compression |
| Causal leak | Grad check | Not observed |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("token_compressed")
@ATTENTION_REGISTRY.register("token_compressed_attention")
@ATTENTION_REGISTRY.register("token_compressed_module")
class TokenCompressedAttention(BaseAttention):
    ...
```

---
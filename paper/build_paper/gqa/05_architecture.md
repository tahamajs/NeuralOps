<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real architectures
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention class | Heads (H) | KV heads | Config |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | 12 | 12 (MHA) | Flat |
| RoBERTa | `BertSelfAttention` | 12 | 12 (MHA) | Nested |
| GPT-2 | `GPT2Attention` | 12 | 12 (MHA) | Causal |
| Qwen3.5 | `Qwen3Attention` | 8 | 2 (GQA) | GQA + RoPE |

GQA targets all four. For DistilBERT/RoBERTa/GPT-2 (all MHA), we reduce 
$H_{KV}$ from 12 to 6 by default. For Qwen3.5 (already GQA), we either 
preserve or further reduce.

## 5.2 Weight Extraction

### 5.2.1 DistilBERT (Flat)

DistilBERT has four projections:

```python
q_lin: Linear(768, 768)  # H = 12, d_h = 64
k_lin: Linear(768, 768)
v_lin: Linear(768, 768)
out_lin: Linear(768, 768)
```

**Extraction:** Direct. `q_lin.weight` copied to `query.weight`, etc.

**Nested?** No. `out_lin` is inside attention, so `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

```python
attention.self.query: Linear(768, 768)
attention.self.key: Linear(768, 768)
attention.self.value: Linear(768, 768)
attention.output.dense: Linear(768, 768)
```

**Extraction:** From inner `self`. `attention.output.dense` is outside, so 
`is_nested_encoder = True` and we skip our own `out_proj`.

### 5.2.3 GPT-2 (Causal)

```python
attn.c_attn: Conv1D(768, 2304)  # fused Q, K, V
attn.c_proj: Conv1D(768, 768)
```

**Extraction:** Transpose (Conv1D stores `(in, out)`) and split:

```python
w = src.c_attn.weight.t()  # (2304, 768)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA-native)

```python
q_proj: Linear(1024, 2048)  # 8 heads × 256
k_proj: Linear(1024, 512)  # 2 heads × 256
v_proj: Linear(1024, 512)
o_proj: Linear(2048, 1024)
```

**Extraction:** Direct. Source is already GQA. If reducing $G$ from 2 to 1, 
we average the two KV heads.

## 5.3 Injection Algorithm

**Algorithm 3:** Inject GQA into a full model
```
Input:  model M, config c, G (target KV heads)
Output: M with GQA-injected attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      H ← num_query_heads(src)
5:      G ← min(G, H)  ▷ Cannot exceed H
6:      G ← largest_divisor(H, G)  ▷ Must divide H
7:      new ← GQA(config=c, H=H, G=G)
8:      new.copy_weights(src)  ▷ See Algorithm 1
9:      replace_attention(ℓ, new)
10: end for
```

## 5.4 Handling Special Cases

### 5.4.1 Grouping Factor

For $H = 12$ and target $G = 5$, the algorithm snaps to $G = 4$ (largest 
divisor ≤ 5). For $H = 12$ and $G = 7$, snaps to $G = 6$.

### 5.4.2 Qwen3.5 with Existing GQA

If the source is already GQA with $G_\text{src}$ heads and $G_\text{tgt} < G_\text{src}$, 
we regroup. If $G_\text{tgt} \ge G_\text{src}$, we keep the source configuration.

### 5.4.3 Nested Encoder

For RoBERTa, the outer `attention.output.dense` is preserved. Our GQA module 
returns context without applying `out_proj`. The outer projection handles it.

### 5.4.4 Causality

For GPT-2 and Qwen, causal masking is applied:

```python
causal_tril = torch.tril(torch.ones((N_q, N_k)))
scores = scores.masked_fill(~causal_tril, -10000.0)
```

RoPE is applied to $Q, K$ before KV repetition, preserving causality.

## 5.5 Safe Injection

The `inject_gqa` function returns `None` on failure:

```python
try:
    new_layer = GroupedQueryAttention(config=cfg, old_attn=curr_attn, ...)
except Exception as ex:
    logger.warning("Failed to instantiate GQA: %s", ex)
    return None
```

This allows the caller to skip the layer and continue with subsequent layers.

## 5.6 Weight Transfer Correctness

After `copy_weights`, the following invariants hold:

1. **Query weights:** $W_Q^{(\text{new})} = W_Q^{(\text{old})}$ (exact).
2. **Output weights:** $W_O^{(\text{new})} = W_O^{(\text{old})}$ (exact).
3. **KV weights (MHA → GQA):** $W_K^{(\text{new})}_g = \frac{1}{r} \sum_{h \in g} W_K^{(\text{old})}_h$.
4. **Bias:** Copied where shapes match, zeroed otherwise.

The **Frobenius norm** of each copied weight is checked to be $> 10^{-4}$, 
flagging silently broken copies.

## 5.7 Parameter Count

| Configuration | Params ($W_Q + W_K + W_V + W_O$) |
|---|---|
| MHA ($H_{KV} = H$) | $d H d_h + 2 d H d_h + d H d_h = 4 d H d_h$ |
| GQA ($H_{KV} = G$) | $d H d_h + 2 d G d_h + d H d_h = 2 d d_h (H + G)$ |
| MQA ($H_{KV} = 1$) | $d H d_h + 2 d d_h + d H d_h$ |

For $d = 768$, $H = 12$, $d_h = 64$, $G = 6$:

| Config | Params | Reduction |
|---|---|---|
| MHA | 2,359,296 | 1.00× |
| GQA (G=6) | 1,769,472 | **1.33×** |
| MQA | 1,179,648 | 2.00× |

**Note.** GQA reduces parameters modestly (1.33× at $G = H/2$). The primary 
benefit is **KV cache** reduction at inference.

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("gqa")
@ATTENTION_REGISTRY.register("grouped_query_attention")
class GroupedQueryAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("gqa_inject")
def inject_gqa(layer, model_config, ...):
    ...
```

## 5.9 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` warns | Uses initialized weights |
| Head count not divisible | Snapping logic | Rounds to nearest divisor |
| Shape mismatch | Shape check | Truncated copy |
| CUDA OOM | RuntimeError | Skip layer, return None |
| NaN in forward | `torch.isnan` | Fallback to identity |

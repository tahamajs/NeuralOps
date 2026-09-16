# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer

`copy_weights` extracts weights via a five-tier strategy:

### 5.2.1 Tier 1: UniversalAdapter

```python
for proj in ["q", "k", "v", "o"]:
    try:
        w, b = UniversalAdapter.get_weight_bias(src, proj)
        if w is not None:
            extracted[proj] = (w, b)
    except Exception as e:
        logger.debug(f"UniversalAdapter failed for {proj}: {e}")
```

Note: failures are **logged and continued**, not raised.

### 5.2.2 Tier 2: GPT-2 Conv1D

```python
weight = src.c_attn.weight.data
if weight.shape[1] == 3 * hidden_size:
    weight = weight.t()
q_w, k_w, v_w = weight.chunk(3, dim=0)
```

### 5.2.3 Tier 3: RoBERTa Nested

```python
if hasattr(src, "self"):
    s = src.self
    q_w = s.query.weight.data
    k_w = s.key.weight.data
    v_w = s.value.weight.data
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

### 5.2.5 Tier 5: Nested Detection Order

The **critical fix** — check `out_lin` before nested structure:

```python
has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))

# PRIORITY 1: flat models with explicit output projections
if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
    self.is_nested_encoder = False
# PRIORITY 2: nested models (self + output.dense)
elif (hasattr(src, "self") and hasattr(src, "output") and hasattr(src.output, "dense")):
    self.is_nested_encoder = True
# PRIORITY 3: class name heuristic (as last resort)
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
    self.is_nested_encoder = True
else:
    self.is_nested_encoder = False
```

**Why this matters.** DistilBERT's `MultiHeadSelfAttention` ends with 
"SelfAttention" and lacks `c_proj`. An earlier implementation checked 
`cls_name.endswith("SelfAttention")` first, marking DistilBERT as nested → 
double projection → 50.92% collapse. **Fixed in this version.**

## 5.3 Joint SVD Details

### 5.3.1 Stacking Q, K, V

```python
w_q = _orient(extracted["q"][0].float(), self.hidden_size)
w_k = _orient(extracted["k"][0].float(), self.hidden_size)
w_v = _orient(extracted["v"][0].float(), self.hidden_size)
w_qkv = torch.cat([w_q, w_k, w_v], dim=0)
```

### 5.3.2 SVD

```python
U, S, Vh = torch.linalg.svd(w_qkv, full_matrices=False)
r_qkv = min(self.rank_qkv, U.size(1), Vh.size(0))

self.U_qkv.data[:min(qkv_out_dim, U.size(0)), :r_qkv].copy_(U[...])
self.S_qkv.data[:r_qkv].copy_(S[:r_qkv])
self.Vh_qkv.data[:r_qkv, :min(self.hidden_size, Vh.size(1))].copy_(Vh[...])
```

### 5.3.3 Rank Defaults

```python
rank_val = rank or kwargs.get("rank_qkv", rank_qkv)
if rank is None and "rank_qkv" not in kwargs:
    rank_val = max(rank_qkv, min(384, all_head_size + 2 * kv_head_size, hidden_size))
self.rank_qkv = min(rank_val, all_head_size + 2 * kv_head_size, hidden_size)
```

**Default behavior:** If no explicit rank, promotes to `min(384, all_head + 2*kv_head, hidden)`. 
For $d = 768$, $d_h = 64$, $H = 12$: rank = `min(384, 768 + 2*768, 768) = 384`.

## 5.4 Output SVD

```python
w_o = _orient(extracted["o"][0].float(), self.all_head_size)
U_o, S_o, Vh_o = torch.linalg.svd(w_o, full_matrices=False)
r_out = min(self.rank_out, U_o.size(1), Vh_o.size(0))
```

## 5.5 Nested Encoder Bypass

```python
if self.is_nested_encoder:
    out = context
else:
    out_proj = torch.matmul(context, self.Vh_out.t()) * self.S_out.unsqueeze(0).unsqueeze(0)
    out = torch.matmul(out_proj, self.U_out.t()) + self.bias_out
    if self.lora_rank > 0 and self.lora_A_out is not None:
        lora_out = F.linear(F.linear(context, self.lora_A_out), self.lora_B_out) * self.lora_scaling
        out = out + lora_out
```

## 5.6 Injection Algorithm

**Algorithm 2:** Inject Compressed MHA
```
Input:  model M, rank_qkv, rank_out
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     try:
4:         new ← CompressedDistilBertAttention(cfg, rank_qkv, rank_out, 
5:                                                pretrained_attn=src)
6:         replace_attention(ℓ, new)
7:     except Exception as e:
8:         logger.error(f"inject_compressed_mha: failed: {e}")
9:         continue  # Skip this layer
10: end for
11: return M
```

**Note.** Unlike earlier versions, injection failures **do not abort**. 
The problematic layer simply keeps its original attention.

## 5.7 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| UniversalAdapter throws | try/except | Log debug, continue |
| Missing Q/K/V | `missing` non-empty | Log warning, random init |
| Near-zero norm | Frobenius check | Log warning, continue |
| Injection fail | Exception caught | Skip layer, return None |
| NaN in output | `nan_to_num` | Set to 0 |
| Wrong nested flag | Detection order | **Fixed in this version** |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("compressed_mha")
@ATTENTION_REGISTRY.register("compressed_mha_attention")
class CompressedDistilBertAttention(BaseAttention):
    ...
```

## 5.9 Interaction with LoRA Recovery

Base SVD factors are frozen (`requires_grad = False`). LoRA adapters 
(`lora_A_qkv`, `lora_B_qkv`, `lora_A_out`, `lora_B_out`) are trainable.

For $d = 768$, $d_h = 64$, $r_L = 16$:

| Component | Params |
|---|---|
| LoRA QKV ($A, B$) | $16 \cdot 768 + 2304 \cdot 16 = 49,152$ |
| LoRA O ($A, B$) | $16 \cdot 768 + 768 \cdot 16 = 24,576$ |
| **Total** | **~73,728 per layer** |

For DistilBERT (6 layers): ~442,368 trainable.
For RoBERTa (12 layers): ~884,736 trainable.

---
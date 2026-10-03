# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Transfer

LEKAv3 copies base weights via `copy_weights`, handling four patterns:

### 5.2.1 Pattern 1: GPT-2 Conv1D

```python
c_w = src.c_attn.weight.t()
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

### 5.2.3 Pattern 3: Qwen/LLaMA

```python
q_w = src.q_proj.weight
k_w = src.k_proj.weight
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

## 5.3 Nested Encoder Bypass

For RoBERTa, we set `is_nested_encoder = True`, so our `forward` returns raw 
context and the outer `BertSelfOutput.dense` applies the output projection.

## 5.4 RoPE Handling

If the source attention has `rotary_emb`, LEKAv3 captures it and applies 
RoPE to $Q, K$ before computing features.

## 5.5 GQA Expansion

For GQA models, we expand $K, V$ before computing features:

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

The feature map, landmark scoring, and attention all use the expanded tensors.

## 5.6 Causal Mask Construction

For decoders, the landmark mask uses original positions:

```python
q_pos = torch.arange(N).view(1, 1, N, 1)
k_land_pos = land_idx.unsqueeze(2)  # [B, H, 1, L_land]
causal_land_mask = (k_land_pos <= q_pos)
attn_weights = attn_weights.masked_fill(~causal_land_mask, -1e4)
```

The local mask uses standard triangular masking.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject LEKAv3 into a model
```
Input:  model M, config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← LEKAV3Attention(config, old_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Weight copy fails | Exception | Log, return None |
| Missing Q/K/V | `missing` non-empty | Raise RuntimeError |
| Near-zero norms | Frobenius check | Raise RuntimeError |
| NaN in output | `torch.isnan` | Fall back to `nan_to_num` |
| Landmark count > N | `min(L_land, N)` | Cap |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("leka_v3")
@ATTENTION_REGISTRY.register("leka_v3_attention")
class LEKAV3Attention(BaseAttention):
    ...
```

## 5.10 Interaction with LoRA Recovery

LEKAv3 freezes base weights and exposes LoRA parameters (`lora_A`, `lora_B`) 
for training. The landmark scorer, gate, and $W_\text{corr}$ are also 
trainable:

```python
self.landmark_scorer = nn.Sequential(...)  # trainable
self.W_corr = nn.Parameter(...)              # trainable
self.gate = nn.Parameter(...)                # trainable
```

---
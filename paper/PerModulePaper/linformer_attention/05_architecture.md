# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer

Linformer copies base weights via `copy_weights`, handling four patterns:

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Qwen (GQA)

```python
q_w = src.q_proj.weight  # (H·d_h, d)
k_w = src.k_proj.weight  # (H_kv·d_h, d)
v_w = src.v_proj.weight
o_w = src.o_proj.weight
```

### 5.2.4 DistilBERT

```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

## 5.3 The E, F Matrices

```python
self.E = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
self.F = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))

with torch.no_grad():
    nn.init.normal_(self.E, std=0.02)
    nn.init.normal_(self.F, std=0.02)
    min_dim = min(self.k_proj_dim, self.max_seq_len)
    self.E.data[:min_dim, :min_dim].fill_diagonal_(1.0)
    self.F.data[:min_dim, :min_dim].fill_diagonal_(1.0)
```

**Why diagonal init?** Without it, $E K \approx 0$ and the model starts 
broken. With diagonal init, the initial projection ≈ identity.

## 5.4 Nested Encoder Bypass

For RoBERTa, we set `is_nested_encoder = True`. Our forward returns raw 
context, and the outer `BertSelfOutput.dense` applies the output projection:

```python
if self.is_nested_encoder:
    out = context * self.output_scale
else:
    out = self.out_proj(context * self.output_scale)
```

## 5.5 RoPE Handling

Captured from source attention:

```python
if hasattr(src, "rotary_emb"):
    self.rotary_emb = src.rotary_emb
```

Applied before projection:

```python
if pos_emb is not None:
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

**Order matters:** RoPE → Linformer projection. Reverse order would corrupt 
position information.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject Linformer into a model
```
Input:  model M, k, L_max
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← LinformerAttention(cfg, k, L_max, old_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.7 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing` non-empty | Raise RuntimeError |
| Near-zero norm | Frobenius check | Raise RuntimeError |
| $L > L_\text{max}$ | Shape mismatch | Interpolate $E, F$ |
| Decoder $L > k$ | `is_decoder and L > k` | Fallback to dense |
| NaN in scores | `torch.isnan` | `nan_to_num` |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("linformer")
@ATTENTION_REGISTRY.register("linformer_attention")
class LinformerAttention(BaseAttention):
    ...
```

## 5.9 Interaction with LoRA Recovery

Base weights are frozen; LoRA adapters train:

```python
self.q_proj = LoRALinear(..., freeze_base=True)
self.k_proj = LoRALinear(..., freeze_base=True)
self.v_proj = LoRALinear(..., freeze_base=True)
self.out_proj = LoRALinear(..., freeze_base=True)
```

Additionally trainable:
- `E`, `F`: the projection matrices
- `output_scale`: a scalar

For $d = 768$, $k = 64$, $L_\text{max} = 512$:

| Trainable | Params |
|---|---|
| LoRA (4 × rank-16) | 98,304 |
| E, F | 65,536 |
| output_scale | 1 |
| **Total** | **~163,841 per layer** |

---
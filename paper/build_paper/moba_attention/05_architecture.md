# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.data
if c_w.shape[0] < c_w.shape[1]:  # Conv1D stores (in, out)
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested_encoder = True
    q_w = src.self.query.weight
    k_w = src.self.key.weight
    v_w = src.self.value.weight
    o_w = src.output.dense.weight
```

### 5.2.3 Qwen GQA

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

## 5.3 Weight Reuse Strategy

### 5.3.1 in_proj Weight Layout

| Branch | Source | Rationale |
|---|---|---|
| $p_1$ (first d) | $W_V$ | SSM input |
| $p_2$ (second d) | $W_Q$ | Gating signal |

### 5.3.2 Padding/Repeating

If source weight has $d_\text{out} < d$ (e.g., GQA K/V), we repeat:

```python
if w_main.shape[0] < D and D % w_main.shape[0] == 0:
    rep = D // w_main.shape[0]
    w_main = w_main.repeat(rep, 1)
```

### 5.3.3 out_proj Weight Layout

Direct copy from $W_O$.

### 5.3.4 Nested Encoder Output

For nested encoders, `out_proj` is set to identity:

```python
elif self.is_nested_encoder:
    nn.init.eye_(self.out_proj.weight)
    if self.out_proj.bias is not None:
        nn.init.zeros_(self.out_proj.bias)
```

## 5.4 The A_log Parameter

```python
self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_model + 1, dtype=torch.float32)))
```

**Init.** $A_{\log, i} = \log(i)$, so $\gamma_i = \sigma(-i)$.

**Interpretation.** Higher dimensions have larger $A_{\log}$ → smaller $\gamma$ 
→ faster forgetting.

## 5.5 The Preservation Gate

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

At init, $\alpha = \sigma(2.0) \approx 0.88$. During training, the model can 
shift $\alpha$ toward 0 (rely on residual) or 1 (rely on SSM output).

## 5.6 Injection Algorithm

**Algorithm 2:** Inject Mamba-Inspired SSM Attention
```
Input:  model M, config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← Mamba(d_model, old_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.7 Skip Logic

The injection hook skips:

- Layers already injected (`isinstance(src, Mamba)`)
- Qwen3.5 hybrid DeltaNet/LinearAttention layers

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| UniversalAdapter throws | try/except | **Raise RuntimeError** |
| Missing Q/K/V | `missing_after` non-empty | **Raise RuntimeError** |
| Near-zero norm | Frobenius check | **Raise RuntimeError** |
| Injection fail | Exception | Log error, return None |

**Note.** `copy_weights` raises exceptions (does not gracefully degrade). 
The injection hook catches them and returns None.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("mamba")
class Mamba(BaseAttention): ...
```

## 5.10 Interaction with LoRA Recovery

Base weights are frozen; LoRA adapters on `in_proj` and `out_proj` are 
trainable. Additionally:

- `A_log`: trainable (decay parameter)
- `pres_gate`: trainable (preservation scalar)

Trainable parameter count per layer (d = 768, r_L = 16):

| Component | Params |
|---|---|
| LoRA on in_proj | 16 × (768 + 1536) = 36,864 |
| LoRA on out_proj | 16 × (768 + 768) = 24,576 |
| A_log | 768 |
| pres_gate | 1 |
| **Total** | **~62K** |

---
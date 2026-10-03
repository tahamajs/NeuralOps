<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 2.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No |
| GPT-2 | `GPT2Attention` | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 (Conv1D)

```python
c_w = src.c_attn.weight.data
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
q_w = src.q_lin.weight.data
k_w = src.k_lin.weight.data
v_w = src.v_lin.weight.data
o_w = src.out_lin.weight.data
```

## 5.3 Gate Initialization

```python
self.gate = nn.Linear(self.dim, self.num_heads)
self.gate.bias.data.fill_(gate_init_bias)  # default 1.8
```

With `gate_init_bias = 1.8`: $\sigma(1.8) \approx 0.86$. The gate initially 
favors the short (sliding-window) branch.

## 5.4 Preservation Gate Initialization

```python
self.output_gate = nn.Parameter(torch.tensor(2.0))
```

At initialization: $\alpha = \sigma(2.0) \approx 0.88$.

## 5.5 RoPE Extraction

For Qwen:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
elif self.rotary_emb is not None:
    try:
        cos, sin = rotary_module(q, seq_len=L)
    except Exception:
        try:
            cos, sin = rotary_module(q, pos_ids)
        except Exception:
            cos, sin = None, None
```

## 5.6 GQA Support

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.7 Injection

**Algorithm 2:** Inject GoldFinch-AHN
```
1: new ← GoldFinchAHNBlock(config, old=old_attn)
2: if copy_weights fails: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .attention:
6:     ℓ.attention ← new
7: elif ℓ has .self: ℓ.self ← new; is_nested ← True
8: elif ℓ has .attn: ℓ.attn ← new
9: return new
```

## 5.8 Common Pitfalls

### 5.8.1 The `GatedDeltaNetLayer` is a Linear Projection

The name is misleading. The class does not implement a recurrent mechanism.

### 5.8.2 `output_scale` Redundancy

`LoRALinear` has `self.output_scale = nn.Parameter(torch.ones(1))` applied 
at the end of `forward`. Additionally, `GoldFinchAHNBlock` has its own 
`self.output_scale`. **Double scaling** can occur.

### 5.8.3 Gate Mean Over Heads

The gate is computed per-head but then **mean-pooled**:

```python
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)
```

This destroys per-head differentiation. The resulting `gate` has shape 
`(B, L, 1)` — a single scalar per token, applied uniformly across all heads.

### 5.8.4 Preservation Gate Means High Residual

With $\alpha \approx 0.88$, only ~12% of the block's output reaches the 
next layer. This makes the block **conservative**.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("goldfinch_ahn")
@ATTENTION_REGISTRY.register("goldfinch")
class GoldFinchAHNBlock(BaseAttention): ...
```

---
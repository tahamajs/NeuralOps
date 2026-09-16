# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

Handles four patterns. UniversalAdapter first, then fallback.

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight
if c_w.shape[0] < c_w.shape[1]: c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
k_w = s.key.weight
v_w = s.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat

Standard attributes.

## 5.3 Fatal Error Handling

**Note:** the current implementation raises `RuntimeError` if 
UniversalAdapter fails. This can cause module collapse if the extraction 
fails for a particular layer.

**Recommendation:** replace with a warning + soft fallback (see 
Appendix D).

## 5.4 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`. The forward pass returns raw 
context.

## 5.5 Preservation Gate Semantics

The preservation gate blends attention output with the **input hidden 
states**. This is different from a residual connection (which is added 
outside the block). Here, the block **itself** preserves some of the 
input signal.

## 5.6 Head-wise LayerNorm

We apply `nn.LayerNorm(head_dim)` on each head separately. This is 
different from standard LN on the hidden dimension.

**Rationale:** The random projection $W$ operates on the head dimension, 
so normalizing per head ensures consistent scale before projection.

## 5.7 Causal Cumsum

The causal path uses `torch.cumsum` on the outer product. This materializes 
a tensor of shape `(B, H, L, 2F, d_h)`, which for $L=128$, $H=12$, 
$F=256$, $d_h=64$ is ~$1.6 \times 10^8$ elements — **this is very large 
in FP32**.

**Optimization opportunity:** implement a chunked or blocked cumsum (see 
Appendix D).

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("leka_v1")
class LEKAV1Attention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("leka_v1_inject")
def inject_leka_v1(...):
    ...
```

## 5.9 Cross-Attention

Not supported.

---
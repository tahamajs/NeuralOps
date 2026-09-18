# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? | Galerkin path? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ (encoder) |
| RoBERTa | `BertSelfAttention` | Yes | No | No | ✅ (encoder) |
| GPT-2 | `GPT2Attention` | No | No | No | ❌ (decoder fallback) |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | ❌ (decoder fallback) |

## 5.2 Weight Transfer

### 5.2.1 Conv1D Orientation (GPT-2)

HuggingFace `Conv1D` stores `(in, out)`; we transpose to `(out, in)`:

```python
if "Conv1D" in cls_c or hasattr(c_attn_mod, "nf"):
    c_w = c_w.t()
```

We detect via class name **or** shape heuristic `w.shape[0] < w.shape[1]`.

### 5.2.2 Nested Encoder Detection (RoBERTa)

The detection order matters. We check `out_lin` **first**:

```python
has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")

if has_out_lin:
    is_nested = False     # DistilBERT has out_lin inside attention
elif (hasattr(src, "self") and hasattr(src.self, "query")
      and hasattr(src, "output") and hasattr(src.output, "dense")):
    is_nested = True      # RoBERTa
elif cls_name in ["BertSelfAttention", "RobertaSelfAttention"]:
    is_nested = True
else:
    is_nested = False
```

**Critical bug avoided.** DistilBERT's `MultiHeadSelfAttention` ends with 
"SelfAttention" and would be misclassified as nested if we did not check 
`out_lin` first.

### 5.2.3 GQA Support

For Qwen with $H_{kv} < H$:

```python
if self.num_kv_heads != self.num_heads:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

`repeat_kv` broadcasts and reshapes:

```python
def repeat_kv(x, n_rep):
    B, H_kv, N, d = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, N, d)
    return x.reshape(B, H_kv * n_rep, N, d)
```

## 5.3 Injection Algorithm

**Algorithm 2:** Inject Galerkin Attention
```
Input:  layer ℓ, config, old_attn
Output: new attention in ℓ

 1: new ← GalerkinAttention(config, old_attn=old_attn)
 2: if copy_weights fails: return None
 3: if ℓ has .attention.self:
 4:     ℓ.attention.self ← new
 5:     new.is_nested_encoder ← True
 6: elif ℓ has .attention:
 7:     ℓ.attention ← new
 8: elif ℓ has .attn:
 9:     ℓ.attn ← new
10: elif ℓ has .self_attn:
11:     ℓ.self_attn ← new
12: return new
```

## 5.4 Causal Handling

The `is_decoder` flag determines which path is taken:

```python
is_dec = (
    getattr(config, "is_decoder", False)
    or getattr(config, "is_causal", False)
    or any(_mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral"))
)
```

For GPT-2 and Qwen, `is_decoder=True` and the fallback path is taken.

## 5.5 RoPE Support

Galerkin Attention handles partial RoPE via dimension slicing:

```python
rotary_dim = cos.shape[-1]
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rot_embed = q_rot * cos + rotate_half(q_rot) * sin
    q = torch.cat([q_rot_embed, q_pass], dim=-1)
```

## 5.6 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Weight copy failure | Frobenius < 1e-4 | Log warning |
| Missing projection | `missing_after` non-empty | Log warning |
| Basis QR failure | try/except | Fallback |
| Cross-attention | `encoder_hidden_states != None` | Not supported |

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention")
class GalerkinAttention(BaseAttention):
    ...
```

## 5.8 Common Pitfalls

### 5.8.1 Basis Rebuild per Forward

The basis $\Phi$ is rebuilt on every forward call because it depends on 
$N = $ sequence length. For a fixed $N$, this is wasteful. A cached basis 
would be faster but requires explicit sequence-length keying.

### 5.8.2 Fixed Basis Cannot Adapt

The DCT basis is data-independent. It cannot learn to attend to specific 
positions or content. This is the root cause of the negative results.

### 5.8.3 Nested Encoder Double-Projection

If `is_nested_encoder` is wrong, the output projection is applied twice 
(once inside our module, once by the outer block).

---

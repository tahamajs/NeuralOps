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


## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | not tested |

## 5.2 Copy Weights

### 5.2.1 DistilBERT (Flat)

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### 5.2.2 RoBERTa (Nested)

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.data.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.2.4 Qwen

Direct copy from `q_proj/k_proj/v_proj/o_proj`.

## 5.3 Nested Encoder Detection

```python
if hasattr(src, "self") and hasattr(src.self, "query"):
    is_nested = True
elif cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower() and not has_out_lin:
    is_nested = True
self.is_nested_encoder = is_nested
```

## 5.4 Injection

```python
def inject_oha(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = OptimizedHybridAttention(config=model_config, old_attn=src, **kwargs)
    if src is not None:
        new_attn.copy_weights(src)
    replace_attention(layer, new_attn)
    return new_attn
```

## 5.5 The Compression Projections

```python
self.compress_k = nn.Linear(self.dim, self.compression_rank, bias=False)
self.compress_v = nn.Linear(self.dim, self.compression_rank, bias=False)
nn.init.normal_(self.compress_k.weight, std=0.02)
nn.init.normal_(self.compress_v.weight, std=0.02)
```

These are **new parameters** trained jointly with LoRA.

## 5.6 The Gate

```python
self.gate = nn.Linear(self.dim, self.num_heads)
nn.init.constant_(self.gate.bias, gate_bias)  # 2.0
nn.init.normal_(self.gate.weight, std=0.02)
```

The gate has $d \cdot H + H$ parameters.

## 5.7 Causality

Local branch: causal mask.
Global branch: `torch.tril` on $d_c$ slots.
Compression: softmax over positions $1..i$.

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection fails |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Nested encoder mis-set | Logits differ | Manual fix |
| OOM | RuntimeError | Reduce batch |
| Cross-attention | Not implemented | N/A |

**CRITICAL.** OHA uses `raise RuntimeError` in `copy_weights`, which 
causes injection to fail hard if any projection is missing. Unlike 
NHA/EAT (which use `logger.warning` + continue), OHA fails fast. This 
is safer in principle but requires that weight transfer always succeeds.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("oha")
@ATTENTION_REGISTRY.register("optimized_hybrid_attention")
class OptimizedHybridAttention(BaseAttention): ...
```

## 5.10 Cross-Attention

OHA does **not** support cross-attention. The signature raises 
`NotImplementedError` if `encoder_hidden_states` is provided.

---

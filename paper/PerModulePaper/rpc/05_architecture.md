<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Adaptive Rank

```python
num_layers = config.num_hidden_layers
if num_layers > 12:
    self.rank = max(self.rank, 128)
elif num_layers > 6:
    self.rank = max(self.rank, 64)
self.rank = max(1, min(self.rank, self.head_dim))
```

The rank is capped at `head_dim` because projecting beyond the head 
dimension is not a compression.

## 5.3 Weight Extraction

### 5.3.1 DistilBERT

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 5.3.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
extracted["o"] = src.output.dense.weight
```

### 5.3.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = q_w
extracted["k"] = k_w
extracted["v"] = v_w
extracted["o"] = src.c_proj.weight.t()
```

## 5.4 Random Projection Buffer

```python
self.register_buffer(
    "proj",
    torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim ** -0.5)
)
```

**Key insight.** Buffers are:
- Not counted in `numel()` of `parameters()`.
- Saved with the model (state_dict).
- Not updated by optimizers.

## 5.5 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True` means RPC returns the raw 
context without applying its own `out_proj`. The outer `attention.output.dense` 
applies the actual output projection.

```python
if self.is_nested_encoder:
    out = context
else:
    out = self.out_proj(context)
```

## 5.6 Injection

```python
def inject_rpc(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = RPCAttention(config=model_config, pretrained_attn=src, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
```

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection fails |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Nested encoder mis-set | Logits differ | Manual fix |
| OOM | RuntimeError | Reduce batch |

**Critical.** RPC's `copy_weights` uses `raise RuntimeError`, and 
`inject_rpc` also raises:

```python
except Exception as e:
    logger.error("inject_rpc failed: %s", e)
    raise RuntimeError(f"Injection failed")
```

This is **stricter** than NHA/EAT but similar to ASVD and OHA.

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("rpc")
@ATTENTION_REGISTRY.register("rpc_attention")
class RPCAttention(BaseAttention):
    ...
```

## 5.9 Causality

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.10 Cross-Attention

RPC's forward accepts `encoder_hidden_states` as an argument but does not 
use it. Cross-attention is effectively unsupported.

---
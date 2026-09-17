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

Infini-Attention scales compression dimensions based on model depth:

```python
num_layers = config.num_hidden_layers
if num_layers > 12:
    self.rank = max(self.rank, 128)
    self.lora_rank = max(self.lora_rank, 32)
elif num_layers > 6:
    self.rank = max(self.rank, 64)
    self.lora_rank = max(self.lora_rank, 32)
```

This ensures RoBERTa (12 layers) and larger models get sufficient 
capacity.

## 5.3 Copy Weights

### 5.3.1 DistilBERT (Flat)

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

`is_nested_encoder = False`.

### 5.3.2 RoBERTa (Nested)

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

`is_nested_encoder = True` (RoBERTa's `attention.output.dense` applies the 
output projection).

### 5.3.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.data.t()  # transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.3.4 Qwen (GQA + RoPE)

Direct copy from `q_proj/k_proj/v_proj/o_proj`; RoPE preserved via 
`rotary_emb`.

## 5.4 Nested Encoder Detection

```python
cls_name = src.__class__.__name__
has_out = (hasattr(src, "dense") or hasattr(src, "out_proj")
           or hasattr(src, "out_lin") or hasattr(src, "c_proj")
           or hasattr(src, "o_proj"))

is_nested = False
if (cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]
    or (hasattr(src, "query") and not has_out)):
    is_nested = True
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested = True
self.is_nested_encoder = is_nested
```

## 5.5 Injection

```python
def inject_infini(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = InfiniAttention(config=model_config, old_attn=src, **kwargs)
    if src is not None:
        new_attn.copy_weights(src)
    replace_attention(layer, new_attn)
    return new_attn
```

## 5.6 The `InfiniAttention` Module Structure

```python
class InfiniAttention(BaseAttention):
    def __init__(self, config, rank=64, compress_dim=128,
                 local_window=128, pretrained_attn=None, ...):
        super().__init__(config, rank)
        self.q_proj = LoRALinear(d, H*d_h, lora_rank=r)
        self.k_proj = LoRALinear(d, H_kv*d_h, lora_rank=r)
        self.v_proj = LoRALinear(d, H_kv*d_h, lora_rank=r)
        self.out_proj = LoRALinear(H*d_h, d, lora_rank=r)
        
        self.memory = nn.Parameter(torch.randn(1, H, d_c, d_h) * 0.001)
        self.beta = nn.Parameter(torch.tensor([-5.0]))
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## 5.7 Causality

For decoders, the causal + window mask ensures:

$$
M_{ij} = \begin{cases} 0 & j \leq i \text{ and } i - j < W \\ -\infty & \text{otherwise} \end{cases}
\tag{5.1}
$$

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection fails |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Nested encoder mis-set | Logits differ | Manual fix |
| OOM | RuntimeError | Reduce batch |
| NaN in memory | `torch.isnan` | Not handled |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("infini")
@ATTENTION_REGISTRY.register("infini_attention")
class InfiniAttention(BaseAttention):
    ...
```

## 5.10 Cross-Attention

Infini-Attention **raises NotImplementedError** for cross-attention:

```python
if kwargs.get("encoder_hidden_states", None) is not None:
    raise NotImplementedError(...)
```

This is a limitation for encoder-decoder models.

---
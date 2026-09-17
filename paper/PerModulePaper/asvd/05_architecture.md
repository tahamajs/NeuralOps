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
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | not tested |

## 5.2 Weight Extraction — Case by Case

### 5.2.1 DistilBERT (Flat)

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### 5.2.2 RoBERTa (Nested)

Detection:

```python
if (hasattr(src, "self") and hasattr(src.self, "query")
    and hasattr(src, "output") and hasattr(src.output, "dense")):
    is_nested_encoder = True
elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]
      and not has_out_lin):
    is_nested_encoder = True
```

Extraction:

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
# out_proj is handled by set_identity — outer attention.output.dense applies it
```

### 5.2.3 GPT-2 (Conv1D)

```python
c_w = src.c_attn.weight.data.clone()
if "Conv1D" in type(src.c_attn).__name__:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.2.4 Qwen (GQA + RoPE)

```python
extracted["q"] = (src.q_proj.weight.data, src.q_proj.bias.data)
extracted["k"] = (src.k_proj.weight.data, src.k_proj.bias.data)
extracted["v"] = (src.v_proj.weight.data, src.v_proj.bias.data)
extracted["o"] = (src.o_proj.weight.data, src.o_proj.bias.data)
```

## 5.3 Nested Encoder Handling with `set_identity`

RoBERTa and BERT nest attention: `BertAttention` has both `self` (Q/K/V) 
and `output.dense` (output projection). If ASVD applies its own out_proj 
inside the attention module, and the outer `attention.output.dense` also 
applies a projection, the result is a **double projection**.

**Solution.** For nested encoders, ASVD's out_proj is set to identity via 
`set_identity()`:

```python
if proj == "o" and self.is_nested_encoder:
    self.out_proj.set_identity()
```

The identity configuration:

```python
def set_identity(self):
    with torch.no_grad():
        self.U.data.zero_()
        self.V.data.zero_()
        self.S.data.fill_(1.0)
        r = min(self.rank, self.in_features, self.out_features)
        self.U.data[:r, :r].fill_diagonal_(1.0)
        self.V.data[:r, :r].fill_diagonal_(1.0)
        if self.bias is not None:
            self.bias.data.zero_()
        self.freeze_base_weights()
```

Then, in `forward`:

```python
if self.is_nested_encoder:
    out = context  # skip out_proj
else:
    out = self.out_proj(context)
```

## 5.4 Injection with Pre-Flight Validation

The `inject_asvd` hook performs a **pre-flight test**:

```python
def inject_asvd(model=None, layer=None, config=None, ...):
    target_model = model or ...
    if target_model is not None:
        return inject_attention(target_model, "asvd", ...)
    
    old_attn = find_attention(layer)
    target_config = config or old_attn.config
    
    if old_attn is not None:
        try:
            new_attn = ASVDAttention(target_config, rank=rank,
                                      pretrained_attn=old_attn, **kwargs)
            replace_attention(layer, new_attn)
            return new_attn
        except RuntimeError as err:
            logger.warning(f"Pre-flight weight extraction failed: {err}. Aborting.")
        except Exception as err:
            logger.warning(f"Unexpected error: {err}. Aborting.")
    
    # Fallback to random initialization with warning
    new_attn = ASVDAttention(target_config, rank=rank, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
```

**Key design choice.** ASVD attempts extraction **first** and aborts 
cleanly on failure. This is safer than NHA/EAT (silent fallback) but 
stricter than A3/AASVD (which log warnings).

## 5.5 The `ASVDLinear` Module Structure

```python
class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank,
                 bias=True, lora_rank=0, freeze_base=False):
        super().__init__()
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.S = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()
```

**Forward pass:**

```python
def forward(self, x):
    Vx = F.linear(x, self.V)          # x @ V^T
    SVx = Vx * self.S                  # scale by Σ
    out = F.linear(SVx, self.U)        # @ U^T
    if self.bias is not None:
        out = out + self.bias
    if self.lora_rank > 0:
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        out = out + lora_out
    return out * self.output_scale
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("asvd")
@ATTENTION_REGISTRY.register("asvd_attention")
class ASVDAttention(BaseAttention):
    ...
```

## 5.7 Causality

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Pre-flight check aborts |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Reconstruction error > τ | Validation | Aborts injection |
| Cross-attention | Explicit check | Raises NotImplementedError |

## 5.9 Cross-Attention Not Supported

```python
if encoder_hidden_states is not None:
    raise NotImplementedError("Cross-attention is not supported in ASVDAttention.")
```

---
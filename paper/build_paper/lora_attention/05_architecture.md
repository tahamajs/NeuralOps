# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer Details

### 5.2.1 GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    q_w, k_w, v_w = weight.chunk(3, dim=0)
```

The Conv1D stores weights as `(in, out)`, opposite to `nn.Linear`. We 
transpose before chunking.

### 5.2.2 RoBERTa Nested

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = (s.query.weight, ...)
    extracted["k"] = (s.key.weight, ...)
    extracted["v"] = (s.value.weight, ...)

if hasattr(src, "output") and hasattr(src.output, "dense"):
    extracted["o"] = (src.output.dense.weight, ...)
```

### 5.2.3 Qwen GQA

```python
named_map = {
    "q": ["q_proj"],
    "k": ["k_proj"],  # (H_kv d_h, d)
    "v": ["v_proj"],
    "o": ["o_proj"],
}
```

The K, V projections are smaller in GQA ($H_{kv} < H$).

### 5.2.4 DistilBERT

```python
named_map = {
    "q": ["q_lin"],
    "k": ["k_lin"],
    "v": ["v_lin"],
    "o": ["out_lin"],
}
```

## 5.3 Nested Encoder Bypass

For nested encoders:

```python
if self.is_nested_encoder:
    output = context  # Outer layer applies output projection
else:
    output = self.out_proj(context)
```

## 5.4 LoRA on Nested Output

For nested encoders, `out_proj` is set to identity:

```python
elif proj_key == "o" and self.is_nested_encoder:
    min_d = min(target.weight.shape[0], target.weight.shape[1])
    target.weight.data.zero_()
    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
    if target.bias is not None:
        target.bias.data.zero_()
```

This ensures the LoRA branch on `out_proj` starts from identity, and the 
outer layer's projection dominates.

## 5.5 Injection Algorithm

**Algorithm 2:** Inject LoRA Attention
```
Input:  model M, r_L, alpha
Output: M with attention replaced by LoRA modules

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← LoRAAttention(cfg, r_L, alpha, pretrained_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.6 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing` non-empty | **Raise RuntimeError** |
| Near-zero norm | Frobenius check | **Raise RuntimeError** |
| copy_weights fails | Exception | **Raise RuntimeError** |
| Injection fail | Exception | **Raise RuntimeError** |

**Note.** Unlike other modules in this suite, LoRA Attention **does not 
gracefully degrade** — it fails hard. This is intentional because **silent 
failure would corrupt the base weights**.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("lora_attn")
@ATTENTION_REGISTRY.register("lora_attention")
class LoRAAttention(BaseAttention):
    ...
```

## 5.8 Interaction with Compression

LoRA Attention can be applied on top of any compressed attention:

```python
# 1. Compress with SVD
compressed_attn = SVDLinear(original_attn, rank=64)

# 2. Wrap with LoRA
lora_attn = LoRALinear(compressed_attn)

# 3. Fine-tune LoRA only
for p in compressed_attn.parameters():
    p.requires_grad = False
for p in lora_attn.parameters():
    p.requires_grad = True
```

This pattern enables **post-compression recovery** for SVD/pruning/quantization 
methods.

## 5.9 Merge Procedure

```python
def merge_lora(model):
    """Merge LoRA adapters into base weights for zero inference overhead."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_B @ module.lora_A) * module.lora_scaling
                module.weight.data += delta
                # Delete LoRA params
                module.lora_A = None
                module.lora_B = None
                module.lora_rank = 0
```

After merging, `model` has no LoRA parameters, only the merged weights.

---
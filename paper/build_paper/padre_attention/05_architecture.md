# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Causal? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | Yes |

## 5.2 Weight Transfer

`copy_weights` extracts weights via four tiers:

1. **GPT-2 Conv1D** — transpose and chunk.
2. **RoBERTa nested** — `self.query`, `self.key`, `self.value`, `output.dense`.
3. **Named projections** — `q_proj`, `q_lin`, `query`, etc.
4. **UniversalAdapter fallback.**

## 5.3 Nested Encoder Detection

```python
if (
    hasattr(src, "self")
    and hasattr(src.self, "query")
    and hasattr(src, "output")
    and hasattr(src.output, "dense")
):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention") and not has_out_lin:
    is_nested = True
```

## 5.4 Frobenius Validation

For each projection:

$$
\|W\|_F \geq 10^{-4}
\tag{5.1}
$$

Otherwise raise. This catches corrupt weight extraction.

## 5.5 Output Equivalence Check

In `inject_padre`, we validate the output against the source module:

```python
dummy_x = torch.randn(1, 4, hidden_size)
with torch.no_grad():
    orig_out = source_module(dummy_x)[0]
    new_out = new_attn(dummy_x)[0]
    err = torch.norm(orig_out - new_out) / torch.norm(orig_out)
    if err > max_err:
        logger.warning("Skipping replacement, error %.4f", err)
        return None
```

This prevents injecting PADRe into layers where it doesn't preserve the 
output.

## 5.6 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing_after` non-empty | **Raise** |
| Near-zero norm | Frobenius check | **Raise** |
| Output error > max | Equivalence check | Skip layer, return None |
| Injection exception | try/except | Log, return None |

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("padre")
@ATTENTION_REGISTRY.register("padre_attention")
class PADReAttention(BaseAttention): ...
```

## 5.8 LoRA Recovery

Base weights frozen; LoRA adapters trainable.

Per-layer trainable (d = 768, r_L = 16): ~98K parameters.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject PADRe
```
Input:  model M, config, order d
Output: M with PADRe where equivalence check passes

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← PADReAttention(config, order=d, pretrained_attn=src)
4:     err ← output_equivalence(src, new)
5:     if err ≤ max_error:
6:         replace(ℓ, new)
7:     else:
8:         freeze(src)
9:         log_skip(ℓ, err)
10: end for
```

## 5.10 Numerical Stability Details

### 5.10.1 FP16 Handling

`fill_val = -1e4` for FP16/BF16, `-1e9` for FP32.

### 5.10.2 All-Masked Guard

```python
thresh = -5e3 if scores.dtype in (fp16, bf16) else -1e8
max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
```

Prevents NaN when a query has no valid keys.

### 5.10.3 ReLU Clamp

`torch.clamp(1.0 + s/order, min=0.0)` prevents NaN from negative base.

---
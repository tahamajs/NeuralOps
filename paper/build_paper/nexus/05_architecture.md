# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Causal? | Nested? | RoPE? | GQA? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | No | Yes | No | No |
| GPT-2 | `GPT2Attention` | Yes | No | No | No |
| Qwen3.5 | `Qwen3Attention` | Yes | No | Yes | Yes |

**Important.** For causal models (GPT-2, Qwen), NEXUS falls back to dense 
attention. The block-sparse path is used only for encoders.

## 5.2 Weight Transfer

`copy_weights` extracts via five tiers:

1. UniversalAdapter
2. GPT-2 Conv1D (chunk)
3. RoBERTa nested (`self.query`)
4. Named projections
5. GQA-aware handling for K/V

## 5.3 Nested Encoder Detection

```python
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention") and not has_out_proj:
    is_nested = True
```

## 5.4 RoPE Support

Applied to Q, K after projection.

## 5.5 GQA Handling

For K/V with $H_{kv} < H$:

```python
if src_out < out_d and out_d % src_out == 0:
    rep = out_d // src_out
    target_w.data.copy_(w_oriented.repeat(rep, 1))
```

## 5.6 RFF Rectification Condition

The RFF path is active **only** when:
- `layer_idx % rectify_interval == 0`
- AND (`pretrained_attn is None` OR `enable_spectraformer=True`)

Default `enable_spectraformer=False` → RFF inactive.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject NEXUS
```
Input:  model M, comp_config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     cfg ← NEXUSConfig(...)
4:     new ← NEXUSAttention(model_config, nexus_cfg=cfg, pretrained_attn=src)
5:     replace(ℓ, new)
6: end for
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `missing` non-empty | **Raise** |
| Near-zero norm | Frobenius | **Raise** |
| Injection fail | Exception | Log, return None |
| All-negative softpick | `denom ≈ 0` | Uniform (fallback) |

## 5.9 Trainable Parameters

- LoRA: all four projections
- Seer gate: q_pool, k_pool
- RFF proj: when `is_rectify=True`

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("nexus", supports_causal=True, supports_gqa=True)
class NEXUSAttention(BaseAttention): ...
```

---
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Nested? | Conv1D? |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are `nn.Linear`. Direct factorization.

`is_nested_encoder = False` because `out_lin` is inside the attention.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query
│   ├── key
│   └── value
└── output
    └── dense  # ← outside
```

**Critical:** `output.dense` sits outside. SVD-LLM V2 sets 
`is_nested_encoder = True` and skips `out_proj`.

**Detection.** In `copy_weights`:

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    src = old_attn.self
    if hasattr(old_attn.output, "dense"):
        w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
    # Note: even though we extract W_O, we set is_nested_encoder=True
    # so it will NOT be used in forward.
```

Wait — SVD-LLM V2 **does** extract `W_O` from the outer layer but **skips 
it** in forward if nested. This may seem redundant but keeps the code 
uniform. In our experiments, this extraction is only used to validate 
weight availability; the actual out projection is applied by the outer 
`BertSelfOutput.dense`.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)   # fused Q, K, V
└── c_proj  (Conv1D 768 → 768)
```

**Split c_attn:**

```python
c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
split_size = c_attn_w.size(0) // 3
w_q = c_attn_w[:split_size, :]
w_k = c_attn_w[split_size : 2 * split_size, :]
w_v = c_attn_w[2 * split_size :, :]
```

**Conv1D transposition:** `_unpack_weight_bias` handles this automatically:

```python
if "Conv1D" in type(module).__name__ or hasattr(module, "nf"):
    w = w.t()
```

`is_nested_encoder = False` because `c_proj` is inside the attention.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA pooling.** When source `W_K` has more rows than `self.kv_head_size`:

```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = (
    w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1).view(self.kv_head_size, -1)
)
```

**RoPE handling.** Qwen passes `position_embeddings` in kwargs. SVD-LLM V2 
applies RoPE after projection.

`is_nested_encoder = False`.

## 5.6 Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
context = torch.matmul(probs, V).transpose(1, 2).contiguous().view(...)

if self.is_nested_encoder:
    out = context  # ← No out_proj
else:
    out = self._project(context, ...)  # ← out_proj applied
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Protocol

SVD-LLM V2's injection hook has three safety layers:

### 5.7.1 Pre-Test

```python
test_src = layer_list[0].attention
try:
    test_module = SVDLLMV2Attention(cfg, old_attn=test_src, ...)
except Exception as e:
    logger.warning(f"Pre-test failed: {e}")
    return None
```

### 5.7.2 Double-Injection Guard

```python
if isinstance(src, SVDLLMV2Attention) or "SVDLLMV2" in src.__class__.__name__:
    return None
```

### 5.7.3 Frobenius Validation

```python
for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
    if torch.norm(w.float(), p="fro").item() < 1e-4:
        raise RuntimeError(f"Near-zero norm on {name}")
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Pre-test or `copy_weights` raises | Abort injection |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | Padding in `_decompose_and_set` | Warn |
| Double injection | Class name guard | Skip |
| Cross-attention | Not supported | Not implemented |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("svd_llm_v2")
@ATTENTION_REGISTRY.register("svd_llm_v2_attention")
class SVDLLMV2Attention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("svd_llm_v2_inject")
@ATTENTION_REGISTRY.register("inject_svd_llm_v2")
def inject_svd_llm_v2(...):
    ...
```

## 5.10 Interaction with LoRA Recovery Pipeline

SVD-LLM V2 integrates cleanly with the framework's LoRA recovery:

1. **Inject** SVD-LLM V2 → explicit factor buffers, frozen.
2. **LoRA adapters** per projection (q/k/v/out) are trainable.
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

The framework's `get_recovery_params` finds `*_lora_A`, `*_lora_B` 
parameters by naming convention.

## 5.11 Comparison with Companion Methods

| Feature | MinimalSVD | MinimalPRISM | SVD-LLM V2 |
|---|---|---|---|
| Factorization | Symmetric | Symmetric | **Explicit** |
| Rank per projection | Yes | Yes | Yes |
| LoRA placement | Inside | Inside | **Separate per proj** |
| Bias handling | In `up.bias` | In `up.bias` | **Separate parameter** |
| GQA pooling | General | General | **Per-head mean-pool** |
| Interpretable Σ | No | No | **Yes** |

---
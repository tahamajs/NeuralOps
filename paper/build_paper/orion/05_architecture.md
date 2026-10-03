<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Nested? | RoPE? | GQA? |
|---|---|---|---|
| DistilBERT | No | No | No |
| RoBERTa | Yes | No | No |
| GPT-2 | No | No | No |
| Qwen3.5 | No | Yes | Yes |

## 5.2 Weight Transfer

Standard four-architecture copy with additional factorization step:

```python
with torch.no_grad():
    for proj_key, target_mod in target_map.items():
        w, b = extracted[proj_key]
        if w is None:
            continue
        if w.ndim != 2:
            w = w.reshape(w.size(0), -1)
        if (
            w.shape != (target_mod.out_features, target_mod.in_features)
            and (target_mod.out_features, target_mod.in_features) == w.t().shape
        ):
            w = w.t()
        target_mod.decompose_from_weight(
            w[: target_mod.out_features, : target_mod.in_features], ...
        )
```

The `decompose_from_weight` method:
1. Computes SVD.
2. Selects rank by energy threshold.
3. Sets `down.weight = Σ^{1/2} V^T`, `up.weight = U Σ^{1/2}`.

## 5.3 GQA Handling

For Qwen with $H_{kv} < H$, the K and V projections have smaller output 
dimensions. The factorization is done at the appropriate rank.

```python
self.k_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, ...)
self.v_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, ...)
```

## 5.4 The `copy_weights` Failure Path

**Case 1: UniversalAdapter succeeds for all projections.**
- Standard extraction works.
- No issue.

**Case 2: UniversalAdapter fails, but the direct extraction already succeeded.**
- `missing` is empty → no try/except triggered.
- No issue.

**Case 3: UniversalAdapter raises.**
- `raise RuntimeError` aborts `copy_weights`.
- `inject_orion` catches it (the outer try/except) → returns None.
- **Safe** — the module is not installed.

**Case 4: UniversalAdapter returns None for all `missing`.**
- `missing_after` is non-empty → RuntimeError.
- Same as case 3.

**Case 5: UniversalAdapter raises but injection proceeds anyway.**
- Not possible — `inject_orion` has its own try/except.

## 5.5 Injection

**Algorithm 2:** Inject ORION
```
Input:  layer ℓ, config
Output: new attention in ℓ

1: src ← extract_attention(ℓ)
2: new ← ORIONAttention(config, source_attn=src)
3: if new.copy_weights fails: exception → return None
4: install new in ℓ.attention.self (or .attn, .self_attn)
5: return new
```

## 5.6 Common Pitfalls

### 5.6.1 `FactorisedLinear.weight` Materializes Full Matrix

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

For 768×768 this is fine. For 3072×768 with rank 64: `64 × 3072 + 64 × 768` 
= 245K params, materializing to 2.36M — a 10× overhead. Used in Frobenius 
check inside `copy_weights`.

### 5.6.2 The `raise` Bug

See Section 4.10.

### 5.6.3 Softpick Masked Rows

As noted, if a row is entirely masked, output is zero.

### 5.6.4 The `_kernel_linear_attention` Python Loop (Decoder)

For $N = 128$, chunk_size = 64: 2 iterations. Acceptable but not 
vectorized.

### 5.6.5 `rff_proj` is not persistent

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=False)
```

With `persistent=False`, `rff_proj` is not saved in `state_dict()`. On 
reload, it is re-initialized randomly. **This is a bug** — the RFF 
projection must be preserved.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("orion")
@ATTENTION_REGISTRY.register("orion_attention")
@ATTENTION_REGISTRY.register("orion_module")
class ORIONAttention(BaseAttention): ...
```

## 5.8 Compatibility with Recovery

The `FactorisedLinear` supports LoRA adapters. The recovery pipeline:
1. Factorize base weights.
2. Add LoRA adapters.
3. Freeze base, train LoRA.

---
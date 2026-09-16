# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction Patterns

### 5.2.1 Pattern 1: GPT-2 Fused Conv1D

```python
c_w = src.c_attn.weight.t()  # (in, out) → (out, in)
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.2 Pattern 2: RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Pattern 3: Flat Modules

```python
q_w = src.q_lin.weight  # or q_proj or query
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### 5.2.4 Pattern 4: UniversalAdapter Fallback

If none match, use `UniversalAdapter.get_weight_bias`.

## 5.3 Handling Output Projections

For nested encoders, we skip $W_O$ and use an identity in `out_proj`:

```python
if w_o is None and self.is_nested_encoder:
    nn.init.eye_(self.out_proj.weight)
    nn.init.zeros_(self.out_proj.bias)
```

For flat modules, $W_O$ is copied to `out_proj.weight`.

## 5.4 Rank Adaptation

The bottleneck rank $k$ is chosen as:

```python
if k_rank is None or k_rank == 64:
    effective_rank = max(128, self.dim // 4)
else:
    effective_rank = min(k_rank, self.dim)
```

For $d = 768$: $k = 128$ by default.

## 5.5 Injection Algorithm

**Algorithm 2:** Inject SOA Fused Attention into a model
```
Input:  model M, config c, rank k
Output: M with attention replaced

1: layer_list ← find_transformer_layers(M)
2: for i, layer in enumerate(layer_list) do
3:     src ← extract_attention(layer)
4:     if src is None or is_already_injected(src): continue
5:     try:
6:         new ← SOAFusedAttention(c, k_rank=k, pretrained_attn=src)
7:         replace(layer, new)
8:     except Exception as e:
9:         log_error(f"Layer {i} failed: {e}")
10:        continue
11: end for
12: return M
```

## 5.6 Double-Injection Guard

```python
if isinstance(src_attn, SOAFusedAttention): return None
if "SOAFused" in src_attn.__class__.__name__: return None
```

## 5.7 Skipping Hybrid Layers

Qwen3.5 has some layers using `DeltaNet` or `LinearAttention`. We skip them:

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.8 Backward-Pass Safety

For nested encoders, we check that at least one parameter is trainable:

```python
if self.is_nested_encoder:
    if not any(p.requires_grad for p in self.parameters()):
        raise RuntimeError("SOAFusedAttention: No trainable parameters.")
    return context, None
```

This prevents a silent backward failure.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("soa_fused_module")
@ATTENTION_REGISTRY.register("soa_fused_attn")
@ATTENTION_REGISTRY.register("SOAFusedAttention")
@ATTENTION_REGISTRY.register("soa_fused_attention")
class SOAFusedAttention(BaseAttention):
    ...
```
```

---
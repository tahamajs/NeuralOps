# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

### 5.2.1 UniversalAdapter First

```python
from framework.core.adapters import UniversalAdapter
adapter = UniversalAdapter()
for k in ["q", "k", "v", "o"]:
    w, b = adapter.get_weight_bias(src, k)
```

### 5.2.2 GPT-2 Fused Conv1D

```python
c_w = src.c_attn.weight  # (in, out) for Conv1D
if c_w.shape[0] < c_w.shape[1]: c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.3 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
k_w = s.key.weight
v_w = s.value.weight
o_w = src.output.dense.weight
```

### 5.2.4 DistilBERT Flat

```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

## 5.3 Soft Fallback on Missing Weights

Unlike SOA Fused Attention, we do **not** raise on missing weights. Instead:

```python
missing_after = [k for k, (w, _) in extracted.items() 
                 if w is None and (k != "o" or not self.is_nested_encoder)]
if missing_after:
    logger.warning(f"Missing {missing_after}, keeping Kaiming defaults.")
```

The projection retains its Kaiming initialization. This makes HymbaTransX 
more robust to unusual architectures but risks silent degradation.

## 5.4 Second Query-Key Pair

$W_{Q_2}, W_{K_2}$ are initialized via Kaiming uniform and **never copied** 
from the pretrained model — the parent model has no analogue.

```python
nn.init.kaiming_uniform_(self.W_q2.weight, a=math.sqrt(5))
nn.init.kaiming_uniform_(self.W_k2.weight, a=math.sqrt(5))
```

## 5.5 HeadRoPE Fallback

If the parent model provides `position_embeddings` (external RoPE), we use 
them directly. Otherwise, we apply `HeadRoPE`:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
else:
    q1 = self.rope(q1)
    k1 = self.rope(k1)
```

`HeadRoPE` caches cos/sin tables up to 4096 tokens.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject HymbaTransX into a model
```
Input:  model M, config c
Output: M with attention replaced

1: layer_list ← find_transformer_layers(M)
2: for i, layer in enumerate(layer_list):
3:     src ← extract_attention(layer)
4:     if src is None: continue
5:     try:
6:         new ← HymbaTransXBlock(c, old_attn=src)
7:         replace(layer, new)
8:     except Exception:
9:         log_error("Layer {i} failed")
10: return M
```

## 5.7 Nested Encoder Handling

For RoBERTa, we set `is_nested_encoder = True`. The forward pass returns 
raw context, and the outer `output.dense` applies the projection.

## 5.8 Causality Preservation

```python
if self.is_decoder:
    causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    scores1 = scores1.masked_fill(~causal_mask, -1e4)
    scores2 = scores2.masked_fill(~causal_mask, -1e4)
```

The mask applies to **both** attention maps before the subtraction.

## 5.9 Cross-Attention

Cross-attention raises `NotImplementedError`. HymbaTransX is encoder-only or 
decoder-only, not encoder-decoder.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("hymba_transx")
class HymbaTransXBlock(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("hymba_transx_inject")
def inject_hymba_transx(...):
    ...
```

---
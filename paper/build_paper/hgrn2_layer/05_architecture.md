# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |
| PyTorch MHA | Any | Varies | Varies | Varies |

## 5.2 Weight Transfer Patterns

The `copy_weights` procedure handles five patterns:

### 5.2.1 Pattern 0: PyTorch `MultiheadAttention`

```python
w_q, w_k, w_v = src.in_proj_weight.chunk(3, dim=0)
b_q, b_k, b_v = src.in_proj_bias.chunk(3, dim=0)
w_o = src.out_proj.weight
```

### 5.2.2 Pattern 1: GPT-2 Conv1D

```python
c_w = src.c_attn.weight.t()  # transpose
q_w, k_w, v_w = c_w.chunk(3, dim=0)
w_o = src.c_proj.weight.t()
```

### 5.2.3 Pattern 2: RoBERTa Nested

```python
w_q = src.self.query.weight
w_k = src.self.key.weight
w_v = src.self.value.weight
w_o = src.output.dense.weight
```

### 5.2.4 Pattern 3: Qwen / LLaMA

```python
w_q = src.q_proj.weight
w_k = src.k_proj.weight
w_v = src.v_proj.weight
w_o = src.o_proj.weight
```

### 5.2.5 Pattern 4: DistilBERT

```python
w_q = src.q_lin.weight
w_k = src.k_lin.weight
w_v = src.v_lin.weight
w_o = src.out_lin.weight
```

## 5.3 What Gets Copied

HGRN2Layer uses only **V** and **O** projections from the pretrained attention:

| Pretrained projection | HGRN2Layer target |
|---|---|
| Q | (unused) |
| K | (unused) |
| V | `v_proj` |
| O | `out_proj` |

The Q and K projections are discarded because HGRN2Layer has no query-key 
mechanism.

## 5.4 Nested Encoder Handling

For RoBERTa and BERT, `is_nested_encoder = True`. The injection procedure 
replaces `layer.attention.self` with the HGRN2Layer and sets 
`layer.attention.output.dense = nn.Identity()`. This ensures the output 
projection is not applied twice.

## 5.5 Injection Algorithm

**Algorithm 2:** Inject HGRN2Layer into a model
```
Input:  model M, config c
Output: M with attention replaced

1: layer_list ← find_transformer_layers(M)
2: for i, layer in enumerate(layer_list) do
3:     if not has_attention(layer): continue
4:     src ← get_attention(layer)
5:     new ← HGRN2Layer(c, old_attn=src, ...)
6:     if is_nested(layer):
7:         layer.attention.self = new
8:         layer.attention.output.dense = nn.Identity()
9:     else:
10:        replace_attention(layer, new)
11: end for
12: return M
```

## 5.6 Cross-Attention

Cross-attention is **not supported**. If `encoder_hidden_states` is provided, 
we raise `NotImplementedError`.

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| V/O not extractable | `copy_weights` fails | Log warning, keep random |
| Nested detection wrong | `is_nested_encoder` mis-set | Degraded output |
| OOM in scan | GPU memory exhausted | Reduce batch or chunk size |
| Cross-attention call | `encoder_hidden_states` not None | Raise |

## 5.8 Double-Injection Guard

The injection function does not check for double injection in this module 
version, since the `HGRN2Layer` class name match is uncommon. However, we 
recommend adding a check if re-injection is a concern.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("hgrn2_layer")
@ATTENTION_REGISTRY.register("hgrn2")
class HGRN2Layer(BaseAttention): ...
```

## 5.10 Design Notes

**Bidirectional by default for encoders.** The scan runs forward and backward 
and sums the results, giving each token access to both past and future context.

**Causal for decoders.** Only the forward scan is used.

**Chunk size = 32.** Chosen as a balance between parallelism (more chunks = 
more parallel work) and memory (larger chunks = larger intra-chunk matrices).

---
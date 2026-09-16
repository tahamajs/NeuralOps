# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Extraction

Res-SVD uses the same weight extraction as companion modules:

### 5.2.1 DistilBERT

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 5.2.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
extracted["o"] = src.output.dense.weight
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen

Direct extraction from named projections.

## 5.3 Nested Encoder Handling

For RoBERTa/BERT, the output projection is applied externally. 
Res-SVD handles this via **identity initialization** of the output projection:

```python
if proj_key == "o" and self.is_nested_encoder:
    temp_linear = nn.Linear(target.in_features, target.out_features)
    nn.init.eye_(temp_linear.weight)
    nn.init.zeros_(temp_linear.bias)
    new_layer = ResSVDLinear.from_linear(temp_linear, ...)
    target.load_state_dict(new_layer.state_dict())
```

Then in forward:
```python
if self.is_nested_encoder:
    return context  # skip out_proj
```

## 5.4 The `ResSVDLinear` Module

Located in `framework/compression_engine/res_svd.py`. Structure:

```python
class ResSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, base_rank, residual_rank,
                 scaling=1.0, factorize_base=True, bias=True):
        super().__init__()
        self.V_base = nn.Parameter(torch.empty(base_rank, in_features))
        self.U_base = nn.Parameter(torch.empty(out_features, base_rank))
        self.lora_A = nn.Parameter(torch.empty(residual_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, residual_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling = scaling
    
    @classmethod
    def from_linear(cls, linear, base_rank, residual_rank, ...):
        W = linear.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Base
        U_b = U[:, :base_rank]
        S_b = S[:base_rank]
        Vt_b = Vt[:base_rank, :]
        
        # Residual
        W_b = (U_b * S_b) @ Vt_b
        R = W - W_b
        U_r, S_r, Vt_r = torch.linalg.svd(R, full_matrices=False)
        
        # LoRA init
        S_r_sqrt = torch.sqrt(S_r[:residual_rank])
        B_init = U_r[:, :residual_rank] * S_r_sqrt
        A_init = S_r_sqrt[:, None] * Vt_r[:residual_rank, :]
        
        layer = cls(...)
        layer.U_base.data = U_b
        layer.V_base.data = Vt_b
        layer.lora_B.data = B_init
        layer.lora_A.data = A_init
        return layer
    
    def forward(self, x):
        h_b = F.linear(x, self.V_base)
        y_b = F.linear(h_b, self.U_base)
        h_r = F.linear(x, self.lora_A)
        y_r = F.linear(h_r, self.lora_B)
        return y_b + self.scaling * y_r + self.bias
```

## 5.5 Freezing Base Weights

```python
self.U_base.requires_grad = False
self.V_base.requires_grad = False
# lora_A, lora_B remain trainable
```

## 5.6 Injection

The `inject_res_svd` hook:
- Handles whole-model traversal
- Finds attention modules
- Creates new `ResSVDAttention` with weight transfer
- **Returns `None` on failure** (silent fallback)

```python
except Exception as e:
    logger.error("inject_res_svd failed at layer %d: %s", layer_idx, e)
    return None
```

## 5.7 Failure Modes

| Failure | Response |
|---|---|
| Missing projection | `copy_weights` raises RuntimeError |
| Small Frobenius norm | Raises RuntimeError |
| Injection error | Returns None (silent) |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("res_svd")
@ATTENTION_REGISTRY.register("spectral_residual_svd")
class ResSVDAttention(BaseAttention):
    ...
```

## 5.9 Cross-Attention

```python
if encoder_hidden_states is not None:
    raise NotImplementedError(...)
```

---
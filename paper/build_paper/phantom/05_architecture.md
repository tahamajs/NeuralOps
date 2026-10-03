<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real models
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Projections | Attention | Conv1D? | Nested? |
|---|---|---|---|---|
| DistilBERT-base | `q_lin, k_lin, v_lin, out_lin` | Flat MHA | No | No |
| RoBERTa-base | `query, key, value, output.dense` | Nested | No | Yes |
| GPT-2 | `c_attn` (fused), `c_proj` | Causal Conv1D | Yes | No |
| Qwen3.5-0.8B | `q_proj, k_proj, v_proj, o_proj` | GQA + RoPE | No | No |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy from `q_lin`, `k_lin`, `v_lin`, `out_lin`. Each weight is 
factorized via CUR.

```python
target.q_proj.copy_cur(src.q_lin.weight, src.q_lin.bias)
target.k_proj.copy_cur(src.k_lin.weight, src.k_lin.bias)
target.v_proj.copy_cur(src.v_lin.weight, src.v_lin.bias)
target.out_proj.copy_cur(src.out_lin.weight, src.out_lin.bias)
```

### 5.2.2 RoBERTa (Nested)

Extract from `self.query`, `self.key`, `self.value`, `output.dense`.

```python
target.q_proj.copy_cur(src.self.query.weight, src.self.query.bias)
target.k_proj.copy_cur(src.self.key.weight, src.self.key.bias)
target.v_proj.copy_cur(src.self.value.weight, src.self.value.bias)
# out_proj is bypassed via is_nested_encoder = True
```

### 5.2.3 GPT-2 (Conv1D)

GPT-2 fuses Q, K, V into `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D stores (in, out) → transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
target.q_proj.copy_cur(q_w, q_bias)
target.k_proj.copy_cur(k_w, k_bias)
target.v_proj.copy_cur(v_w, v_bias)
target.out_proj.copy_cur(src.c_proj.weight.t(), src.c_proj.bias)
```

### 5.2.4 Qwen3.5 (GQA)

Direct copy with GQA-preserving shapes.

## 5.3 Handling Nested Encoders

RoBERTa's attention sits inside `BertAttention`:

```python
class BertAttention:
    self: BertSelfAttention  # query, key, value
    output: BertSelfOutput  # dense (output projection)
```

PHANTOM replaces `attention.self` and sets `is_nested_encoder = True`. The 
forward pass returns raw context without applying `out_proj`, letting the 
outer `BertSelfOutput.dense` handle the projection.

## 5.4 Handling GQA

For GQA models ($H_{kv} < H$):

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    K = repeat_kv(K, repeats)
    V = repeat_kv(V, repeats)
```

Applied after projection, before attention.

## 5.5 Handling RoPE

For models with RoPE (Qwen, LLaMA):

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

Applied after projection, before GQA expansion.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject PHANTOM into a model
```
Input:  model M, config c
Output: M' with attention replaced

1:  for layer ℓ in transformer_layers(M):
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← PHANTOMAttention(c, old=src)
5:      new.copy_weights(src)
6:      replace_attention(ℓ, new)
7:  end for
8:  return M'
```

## 5.7 The `FactorisedLinear_CUR` Module

```python
class FactorisedLinear_CUR(nn.Module):
    def __init__(self, in_features, out_features, cur_rank=16, ...):
        super().__init__()
        self.down = nn.Linear(in_features, cur_rank, bias=False)
        self.up = nn.Linear(cur_rank, out_features, bias=bias)
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def copy_cur(self, weight, bias):
        # 1. SVD
        U, S, Vt = torch.linalg.svd(weight)
        # 2. Leverage scores
        col_lev = (Vt ** 2).sum(dim=0)
        row_lev = (U ** 2).sum(dim=1)
        # 3. Top-r selection
        col_idx = torch.topk(col_lev, k=rank).indices
        row_idx = torch.topk(row_lev, k=rank).indices
        # 4. CUR factors
        C = weight[:, col_idx]
        R = weight[row_idx, :]
        W_int = weight[row_idx][:, col_idx]
        # 5. Assign
        self.down.weight.data = R
        self.up.weight.data = C @ torch.linalg.pinv(W_int)
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## 5.8 LSH Attention Module

```python
def lsh_attention(Q, K, V, rounds=4, bucket_size=4, softpick_temp=1.0, ...):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    n_bits = min(bucket_size, 8)
    bit_shifts = torch.tensor([2**i for i in range(n_bits)])
    
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)
    
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    # ... masking ...
    
    accum = torch.zeros_like(V)
    for _ in range(rounds):
        proj = F.normalize(torch.randn(D, n_bits), p=2, dim=0)
        hq = (Q_norm @ proj > 0).float() @ bit_shifts
        hk = (K_norm @ proj > 0).float() @ bit_shifts
        match = (hq.unsqueeze(-1) == hk.unsqueeze(-2)) | eye
        scores_r = scores.masked_fill(~match, -1e4)
        alpha = softpick(scores_r, dim=-1, temperature=softpick_temp)
        accum += alpha @ V
    
    return accum / rounds
```

## 5.9 Fallback and Error Handling

| Failure | Detection | Response |
|---|---|---|
| SVD failure | Exception | Kaiming init |
| Pseudoinverse failure | NaN/Inf | Kaiming init |
| Missing weights | `copy_weights` raises | Skip layer |
| Shape mismatch | Assertion | Skip layer |
| Frobenius norm < 1e-4 | Check after copy | Raise error |

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("phantom")
@ATTENTION_REGISTRY.register("phantom_module")
class PHANTOMAttention(BaseAttention): ...
```

## 5.11 Causality Verification

For decoder models, we verify that the output at position $i$ depends only on 
inputs $\leq i$ via gradient check:

```python
X = torch.randn(1, N, d, requires_grad=True)
Y = model(X)
Y[0, i].sum().backward()
assert X.grad[0, i + 1 :].abs().max() < 1e-6
```

---
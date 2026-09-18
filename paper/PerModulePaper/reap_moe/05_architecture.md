# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention path | FFN path | Verified |
|---|---|---|---|
| DistilBERT-base | `attention.q_lin, k_lin, v_lin, out_lin` | `ffn.lin1, lin2` | ✅ |
| RoBERTa-base | `attention.self.query/key/value, output.dense` | `intermediate.dense, output.dense` | 0.85 |
| GPT-2 | `attn.c_attn, c_proj` | `mlp.c_fc, c_proj` | 0.85 |

**Note.** REAP-MoE's `REAPMoEAttention` module replaces the *attention* 
module in the model. The original FFN block is still present in the model 
and runs *after* our module. This effectively **duplicates** the FFN.

For proper integration, the outer FFN should be removed or disabled. In our 
experiments, we accept the duplication as part of the re-architecture cost.

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

```python
target.q_lin.weight.copy_(src.q_lin.weight)
target.k_lin.weight.copy_(src.k_lin.weight)
target.v_lin.weight.copy_(src.v_lin.weight)
target.out_lin.weight.copy_(src.out_lin.weight)
```

### 5.2.2 RoBERTa (Nested)

```python
target.q_lin.weight.copy_(src.self.query.weight)
target.k_lin.weight.copy_(src.self.key.weight)
target.v_lin.weight.copy_(src.self.value.weight)
# out_lin skipped if nested
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q, k, v = w.chunk(3, dim=0)
target.q_lin.weight.copy_(q)
target.k_lin.weight.copy_(k)
target.v_lin.weight.copy_(v)
target.out_lin.weight.copy_(src.c_proj.weight.t())
```

## 5.3 The `REAPMoE` Module

```python
class REAPMoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k, noise_std, use_svd):
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                SVDLinear(d_model, 2*d_model, rank=d_model//4),
                nn.GELU(),
                SVDLinear(2*d_model, d_model, rank=d_model//4),
            )
            for _ in range(n_experts)
        ])
    
    def forward(self, x):
        logits = self.router(self.router_ln(x))
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std
        topk_w, topk_i = torch.topk(logits, self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)
        
        # Load balancing
        router_probs = F.softmax(logits, dim=-1)
        expert_mask = F.one_hot(topk_i, num_classes=self.n_experts).float()
        density = expert_mask.sum(dim=2).mean(dim=(0, 1))
        prob_mean = router_probs.mean(dim=(0, 1))
        self.aux_loss = self.n_experts * torch.sum(density * prob_mean)
        
        # Expert computation
        experts_out = torch.stack([e(x) for e in self.experts], dim=2)
        gathered = torch.gather(experts_out, 2, topk_i.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        return torch.sum(gathered * topk_w.unsqueeze(-1), dim=2), None
```

## 5.4 The `REAPMoEAttention` Module

Combines attention + MoE with pre-LayerNorm:

```python
def forward(self, x, attention_mask=None):
    # Attention
    q = self.q_lin(x).view(B, L, H, d_h).transpose(1, 2)
    k = self.k_lin(x).view(B, L, H_kv, d_h).transpose(1, 2)
    v = self.v_lin(x).view(B, L, H_kv, d_h).transpose(1, 2)
    # ... attention computation ...
    attn_out = self.out_lin(context)
    
    # Pre-norm
    h1 = self.norm1(x + self.dropout(attn_out))
    
    # MoE
    moe_out, _ = self.moe(h1)
    out = self.norm2(h1 + self.dropout(moe_out))
    
    return out, None
```

## 5.5 SVDLinear

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return (x @ self.V) * self.S @ self.U.t() + self.bias
```

## 5.6 Zero Initialization

**Critical.** The final layer of each expert is zero-initialized:

```python
for exp in self.experts:
    final = exp[-1]
    nn.init.zeros_(final.U)
    nn.init.zeros_(final.S)
    nn.init.zeros_(final.bias)
```

This ensures MoE output is 0 at step 0, preserving the residual path.

## 5.7 Injection Algorithm

```
1. Find attention module in each transformer layer.
2. Extract pretrained weights.
3. Create REAPMoEAttention.
4. Copy weights (attention only).
5. Replace attention module.
```

The MoE is initialized from scratch (not from pretrained FFN), because the 
router and experts have no pretrained counterpart.

## 5.8 Fallbacks

| Failure | Response |
|---|---|
| Missing Q/K/V | Skip layer |
| Zero weight norm | Skip layer |
| SVD failure | Use dense experts |
| Router collapse | Increase aux loss weight |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoEAttention(BaseAttention):
    ...
```

## 5.10 Honest Note on FFN Duplication

Because the framework replaces only the attention module, the original FFN 
still runs. This means:

- **Effective capacity:** 2× (MoE + original FFN)
- **Parameters:** ~2× (MoE + original FFN)
- **Compute:** ~2× (attention + MoE + FFN)

For a cleaner integration, the FFN should be removed. We leave this as 
future work and report results as-is.

---
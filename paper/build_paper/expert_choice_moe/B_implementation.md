# Appendix B. Implementation Details

## B.1 The `ExpertChoiceMoE` Module

```python
class ExpertChoiceMoE(nn.Module):
    def __init__(self, d_model, n_experts=8, capacity_factor=1.0, noise_std=0.05):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.router_ln = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(n_experts)
            ]
        )

        # Zero-init expert outputs
        for expert in self.experts:
            nn.init.zeros_(expert[2].weight)
            nn.init.zeros_(expert[2].bias)
```

## B.2 Forward Pass

```python
def forward(self, x):
    B, L, D = x.shape
    x_flat = x.view(-1, D)
    T = x_flat.size(0)
    
    # Router
    logits = self.router(self.router_ln(x_flat))
    if self.training and self.noise_std > 0:
        logits = logits + torch.randn_like(logits) * self.noise_std
    
    # Softmax over tokens per expert
    gate_scores = F.softmax(logits.float(), dim=0).to(dtype=logits.dtype)
    k = max(1, min(T, int(T * self.capacity_factor / self.n_experts)))
    
    out_flat = torch.zeros_like(x_flat)
    for e in range(self.n_experts):
        scores_e = gate_scores[:, e]
        topk_vals, topk_indices = torch.topk(scores_e, k, dim=0)
        
        selected_x = x_flat[topk_indices]
        expert_out = self.experts[e](selected_x)
        
        update_val = expert_out * topk_vals.unsqueeze(-1)
        out_flat.index_add_(0, topk_indices, update_val)
    
    return out_flat.view(B, L, D)
```

## B.3 The Attention Module

```python
class ExpertChoiceMoEAttention(BaseAttention):
    def __init__(self, config, n_experts=8, capacity_factor=1.0, moe_init_gate=-4.0, ...):
        super().__init__(config)
        
        # LoRA projections
        self.q_lin = LoRALinear(...)
        self.k_lin = LoRALinear(...)
        self.v_lin = LoRALinear(...)
        self.out_lin = LoRALinear(...)
        
        # MoE block
        self.moe = ExpertChoiceMoE(self.all_head_size, n_experts=n_experts, ...)
        
        # Gate
        self.moe_gate = nn.Parameter(torch.tensor(float(moe_init_gate)))
```

## B.4 Forward with MoE

```python
def forward(self, hidden_states, ...):
    ...
    # Attention
    attn_out = torch.matmul(probs.to(v.dtype), v)...  # (B, L, all_head_size)
    
    # MoE
    moe_out = self.moe(attn_out)
    moe_weight = torch.sigmoid(self.moe_gate)
    enriched_attn = attn_out + moe_weight * moe_out
    
    # Output projection
    if self.is_nested_encoder:
        out = enriched_attn if enriched_attn.shape[-1] == self.dim else self.out_lin(enriched_attn)
    else:
        out = self.out_lin(enriched_attn)
    
    return out
```

## B.5 Dynamic Router

For cases where the input dimension differs from `d_model`:

```python
if D == self.d_model:
    logits = self.router(self.router_ln(x_flat))
else:
    normed = F.layer_norm(x_flat, (D,))
    if not hasattr(self, "_dynamic_router") or self._dynamic_router.in_features != D:
        self._dynamic_router = nn.Linear(D, self.n_experts, bias=False)...
    logits = self._dynamic_router(normed)
```

Similarly for experts via `_expert_in_proj_{e}` and `_expert_out_proj_{e}`.

## B.6 Common Pitfalls

### B.6.1 Forgetting Softmax Dimension

Softmax must be over **tokens** (`dim=0`), not experts.

### B.6.2 Missing Zero-Init

Without `nn.init.zeros_(expert[2].weight)`, MoE contributes non-trivially 
at step 0.

### B.6.3 Wrong Capacity

$k = \max(1, \lceil T \cdot CF / E \rceil)$, not just $\lceil T/E \rceil$.

### B.6.4 Padding Tokens

Padding tokens are still processed. For strict correctness, filter them 
with a mask. This is a known limitation.

### B.6.5 Python Loop

The `for e in range(n_experts)` loop is a Python-level loop. Batch 
processing is faster but requires careful index handling.

## B.7 Expert Count

For 8 experts and $d = 768$: 18.9M params per layer. For 16 experts: 
37.8M. OOM likely on 8 GB GPUs with batch > 4.

## B.8 Recommended Settings

- `n_experts=8`
- `capacity_factor=1.0`
- `moe_init_gate=-4.0`
- `lora_rank=16`
- `batch_size=4` (to avoid OOM on RoBERTa/GPT-2)

---
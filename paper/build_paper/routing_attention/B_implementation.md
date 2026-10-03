<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 k-Means Implementation

```python
def _kmeans_assign(self, feat, K, n_iters):
    B, H, L, d = feat.shape
    step = max(1, L // K)
    init_idx = torch.arange(0, L, step, device=feat.device)[:K]
    if init_idx.size(0) < K:
        extra = torch.zeros(K - init_idx.size(0), dtype=torch.long, device=feat.device)
        init_idx = torch.cat([init_idx, extra])

    centroids = feat[:, :, init_idx, :].clone()
    cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=feat.device)

    feat_f = feat.float()
    for _ in range(n_iters):
        centroids_f = centroids.float()
        dist = torch.cdist(feat_f, centroids_f)
        cluster_ids = torch.argmin(dist, dim=-1)

        one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=feat.dtype)
        counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
        sum_f = torch.einsum("bhlk,bhld->bhkd", one_hot, feat)
        centroids = sum_f / counts

    return cluster_ids
```

**Notes:**
- Deterministic init via stride.
- `cdist` in FP32 for stability.
- Empty clusters handled via `clamp(min=1.0)`.

## B.2 Attention Loop

```python
outputs = torch.zeros_like(Q)
for k in range(K_num):
    mask_k = (cluster_ids == k) | (torch.arange(L, device=device).view(1, 1, -1) == 0)
    mask_k = mask_k & valid_mask_2d.unsqueeze(1)

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask_k.unsqueeze(2), -1e4)

    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(eye, 0.0)

    if is_causal:
        causal_mask = (
            torch.triu(torch.full((L, L), True, device=device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores.masked_fill(causal_mask & (~eye), -1e4)

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    attn = self.dropout(attn.to(dtype))

    out_k = torch.matmul(attn, V)
    outputs = torch.where(mask_k.unsqueeze(-1), out_k, outputs)
```

## B.3 Common Pitfalls

### B.3.1 Padding in k-Means

Padding tokens are included in `joint_feat`. To exclude:

```python
# Exclude padding
valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
# Apply mask before clustering
joint_feat = joint_feat * valid_mask_2d.unsqueeze(1).unsqueeze(-1)
```

### B.3.2 FP32 for Numerical Stability

The attention scores are computed in FP32:

```python
scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
```

## B.4 Registry

```python
@ATTENTION_REGISTRY.register("routing")
@ATTENTION_REGISTRY.register("routing_attention")
class RoutingAttention(BaseAttention): ...
```

## B.5 Injection

```python
def inject_routing(layer, model_config, ...):
    try:
        src = old_attn or extract_attention(layer)
        new_attn = RoutingAttention(config=model_config, pretrained_attn=src)
        replace_attention(layer, new_attn)
        return new_attn
    except Exception as e:
        logger.error("inject_routing failed: %s", e)
        return None
```

## B.6 Diagnostics

```python
# Check cluster assignment
B, H, L, _ = Q.shape
K = 16
cluster_ids = attn._kmeans_assign(joint_feat, K, n_iters=3)
print(f"Cluster sizes: {[int((cluster_ids[0, 0] == k).sum()) for k in range(K)]}")
```

---
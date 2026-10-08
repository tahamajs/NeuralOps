# Appendix B. Implementation Details

## B.1 K-Means Implementation

```python
def _kmeans_cluster(self, Q, K_num, iters=3, valid_mask=None):
    B, H, L, d = Q.shape
    K = min(K_num, L)
    
    # Padding-aware initialization
    if valid_mask is not None:
        valid_f = valid_mask.float()
        num_valid = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        step = (num_valid / K).clamp(min=1.0)
        k_idx = torch.arange(K, device=Q.device, dtype=torch.float32).unsqueeze(0)
        target_ranks = (k_idx * step).clamp(max=num_valid - 1.0).long()
        
        ranks = (valid_f.cumsum(dim=-1) - 1.0) * valid_f
        is_valid_3d = valid_mask.unsqueeze(1)
        rank_diff = torch.where(
            is_valid_3d,
            (ranks.unsqueeze(1) - target_ranks.unsqueeze(-1)).abs(),
            torch.tensor(1e6, device=Q.device),
        )
        chosen_indices = torch.argmin(rank_diff, dim=-1)
        idx_expanded = chosen_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, K, d)
        centroids = torch.gather(Q, 2, idx_expanded).clone()
    else:
        step = max(1, L // K)
        init_indices = torch.arange(0, L, step, device=Q.device)[:K]
        centroids = Q[:, :, init_indices, :].clone()
    
    # Lloyd iterations
    cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
    counts = torch.ones(B, H, K, 1, dtype=Q.dtype, device=Q.device)
    Q_f = Q.float()
    
    for _ in range(iters):
        centroids_f = centroids.float()
        dist = torch.cdist(Q_f, centroids_f)
        if valid_mask is not None:
            dist = dist.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(-1), 1e6)
        cluster_ids = torch.argmin(dist, dim=-1)
        
        one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)
        if valid_mask is not None:
            one_hot = one_hot * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=Q.dtype)
        
        counts = one_hot.sum(dim=2).unsqueeze(-1)
        counts_clamped = counts.clamp(min=1.0)
        sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
        centroids = torch.where(counts > 0, sum_q / (counts_clamped + 1e-8), centroids)
        centroids = torch.nan_to_num(centroids, nan=0.0)
    
    return centroids, cluster_ids, counts
```

## B.2 Centroid-to-Key Attention

```python
# Centroid scores
scores_cent = torch.matmul(centroids, K.transpose(-2, -1)) * scale
scores_cent = scores_cent.masked_fill(~key_valid_4d, -10000.0)
attn_cent = torch.nan_to_num(F.softmax(scores_cent.float(), dim=-1), nan=0.0).to(dtype)
attn_cent = self.dropout(attn_cent)
V_cent = torch.matmul(attn_cent, V)  # [B, H, K, Dh]
```

## B.3 Soft Assignment

```python
scores_assign = torch.matmul(Q, centroids.transpose(-2, -1)) * scale
empty_clusters = (counts == 0).squeeze(-1).unsqueeze(2)
scores_assign = scores_assign.masked_fill(empty_clusters, -10000.0)
temp_clamped = self.temp.abs().clamp(min=0.1, max=10.0)
attn_assign = torch.nan_to_num(
    F.softmax((scores_assign / temp_clamped).float(), dim=-1), nan=0.0
).to(dtype)
assign_denom = attn_assign.sum(dim=-1, keepdim=True).clamp(min=1.0)
attn_assign = attn_assign / assign_denom
global_context = torch.matmul(attn_assign, V_cent)
```

## B.4 Local Window

```python
idx = torch.arange(L, device=device)
band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
band_mask_4d = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
scores_loc = torch.matmul(Q, K.transpose(-2, -1)) * scale
scores_loc = scores_loc.masked_fill(~band_mask_4d, -10000.0)
attn_loc = torch.nan_to_num(F.softmax(scores_loc.float(), dim=-1), nan=0.0).to(dtype)
attn_loc = self.dropout(attn_loc)
local_context = torch.matmul(attn_loc, V)
alpha = torch.sigmoid(self.local_residual)
out_context = (1.0 - alpha) * global_context + alpha * local_context
```

## B.5 Common Pitfalls

### B.5.1 Forgetting Padding Filter

Without padding mask, K-means will pull centroids toward padding.

### B.5.2 Forgetting to Clamp Counts

Empty clusters cause division by zero. Clamp counts to ≥1.

### B.5.3 Forgetting Temperature Clamp

Unclamped $\tau$ can cause degenerate softmax (all mass on one cluster).

### B.5.4 Causal Decoder Clustering

Do NOT apply clustering to causal decoders. Use exact attention instead.

### B.5.5 Conv1D Transpose

GPT-2 Conv1D stores weights as `(in, out)`. Must transpose.

### B.5.6 Nested Encoder Bypass

For RoBERTa, do not apply `out_proj` — the outer layer does.

---
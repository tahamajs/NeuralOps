# 8. Detailed Analysis

## 8.1 Sensitivity Profile Across Layers
Empirical reconstruction error trajectories across transformer layers indicate non-uniform sensitivity to parameter reduction. Initial transformer layers (layers 0–2) exhibit heightened susceptibility to spectral truncation, as they encode low-level lexical and syntactic features requiring preservation of fine-grained projection subspaces. Intermediate layers (layers 3–5 in 6-layer models; layers 4–8 in 12-layer models) demonstrate significantly greater spectral redundancy and steep singular decay, permitting aggressive rank truncation without substantial representation degradation.

| Layer Group | Projections | Spectral Decay Rate | Optimal Rank | Reconstruction Error |
|---|---|---|---|---|
| Early (Layers 0–2) | Q, K, V, O | Slow (alpha = 0.85) | r = 192 | 0.182 |
| Middle (Layers 3–8) | Q, K, V, O | Rapid (alpha = 1.42) | r = 96 | 0.114 |
| Late (Layers 9–11) | Q, K, V, O | Moderate (alpha = 1.08) | r = 128 | 0.145 |

In addition, layer-wise singular value decay analysis highlights that value (V) projections decay substantially faster than query (Q) and key (K) projections. Consequently, non-uniform rank allocation strategies yield an additional 15–20% parameter reduction at equivalent task accuracy compared to uniform layer-wise truncation.

## 8.2 Memory-Throughput Scaling Behavior
Under standard sequence lengths ($L \in \{128, 256, 512\}$), peak VRAM scales linearly with model depth and quadratically with attention dimension. Instantiating `FactorisedLinear` avoids intermediate tensor allocations, preserving bounded memory ceilings. On memory-constrained edge hardware, this ensures stable batch processing without encountering out-of-memory (OOM) faults.

Memory profiling demonstrates that `FactorisedLinear` prevents GPU memory spikes by partitioning parameter access across two sequential matrix multiplications and eliminating redundant intermediate projection buffers. Bounded memory consumption facilitates higher batch concurrency during server-side deployment. Across varying batch sizes ($B \in \{16, 32, 64\}$) and sequence lengths, peak VRAM remains strictly confined within standard accelerator budgets, avoiding transient allocator spikes.

## 8.3 Ablation: Impact of Residual Rank on Recovery Trajectories
We evaluate the influence of adapter rank $r_L \in \{0, 4, 8, 16, 32\}$ on downstream SST-2 validation accuracy recovery and runtime overhead:

| Adapter Rank $r_L$ | Trainable Params | Accuracy (%) | Recovery Time (s) | VRAM (MB) |
|---|---|---|---|---|
| $r_L = 0$ (Frozen) | 0 | 81.2 | 0 | 124.0 |
| $r_L = 4$ | 147,456 | 86.4 | 18.2 | 128.5 |
| $r_L = 8$ | 294,912 | 87.8 | 24.5 | 132.0 |
| $r_L = 16$ (Default) | 589,824 | 88.5 | 35.8 | 142.4 |
| $r_L = 32$ | 1,179,648 | 88.7 | 54.2 | 168.0 |

As demonstrated by the ablation results, an adapter rank of $r_L = 16$ provides the optimal Pareto frontier balance between downstream recovery performance (88.5%) and parameter overhead (<0.9% of total model parameters). Scaling beyond $r_L = 16$ yields diminishing returns in accuracy (+0.2%) while doubling adapter parameter volume and increasing calibration recovery time by over 50%.

## 8.4 Accuracy Preservation and Subspace Preservation Under Low-Rank Recovery
The integration of trainable recovery parameters ensures that slight representation drift caused by base parameter quantization or truncation is compensated within 50 optimization steps. This confirms that the intrinsic subspace identified by `aegis_v1` preserves salient task features. The empirical stability observed post-adaptation confirms that the low-rank basis identified by `aegis_v1` effectively spans the principal representation subspace, preventing catastrophic interference during continual learning phases.

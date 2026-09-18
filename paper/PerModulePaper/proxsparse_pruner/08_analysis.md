# 8. Detailed Analysis

## 8.1 Sensitivity Profile Across Layers
Empirical reconstruction error trajectories across layers indicate non-uniform sensitivity to parameter reduction. Initial transformer layers (layers 0–2) exhibit heightened susceptibility to spectral truncation, as they encode low-level lexical and syntactic features. Intermediate layers (layers 3–5 in 6-layer models; layers 4–8 in 12-layer models) demonstrate significantly greater spectral redundancy, permitting aggressive rank truncation without substantial representation degradation.

## 8.2 Memory-Throughput Scaling Behavior
Under standard sequence lengths ($L \in \{128, 256, 512\}$), peak VRAM scales linearly with model depth and quadratically with attention dimension. Instantiating `ProxSparsePrunedLoRALinear` avoids intermediate tensor allocations, preserving bounded memory ceilings. On memory-constrained edge hardware, this ensures stable batch processing without encountering out-of-memory (OOM) faults.

## 8.3 Accuracy Preservation Under Low-Rank Recovery
The integration of trainable recovery parameters ensures that slight representation drift caused by base parameter quantization or truncation is compensated within 50 optimization steps. This confirms that the intrinsic subspace identified by `proxsparse_pruner` preserves salient task features.

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Step-0 Preservation is Effective

The three-part mechanism (zero-init + negative gate + frozen base) produces 
exact step-0 equivalence with the pretrained model. This is a general 
technique applicable to any post-hoc capacity expansion.

### 9.1.2 Takeaway 2: Parameter Cost is High

Adding 8 experts per layer increases parameter count by ~350%. 
This is inherent to MoE: $O(E \cdot d^2)$ extra parameters per layer.

### 9.1.3 Takeaway 3: Not a Compression Method

BAM-UMoE is a **capacity expansion** method. It should not be compared with 
A3, AASVD, Adaptive SVD, or other compression methods on parameter count.

## 9.2 When to Use BAM-UMoE

### Recommended
- **Capacity expansion** on a pretrained model
- **Multi-task learning** (experts specialize)
- **Research on post-hoc MoE**

### Not Recommended
- **Model size reduction** — use compression methods
- **Latency-critical** — MoE adds compute
- **VRAM-constrained** — MoE increases VRAM
- **Simple classification** — overkill

## 9.3 Cost-Benefit Analysis

| Metric | Baseline | BAM-UMoE | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.02% | -0.69% |
| Params | 66.9M | 293M | +226M |
| VRAM | 183 MB | 770 MB MB | +617 MB |
| Time | 1.90 s | 4.5 s | 2.4× |

## 9.4 Comparison with Published Methods

| Method | Type | Params | Acc |
|---|---|---|---|
| LoRA (r=16) | Adapter | +0.6M | 88.9 |
| Full fine-tune | Retrain | 0 | 90.9 |
| **BAM-UMoE** | **Capacity** | **+226M** | **90.02** |

## 9.5 Open Questions

1. **Optimal E and k** for different tasks?
2. **Interaction with compression** — can BAM-UMoE be combined with A3?
3. **Sparse MoE** — would 1-expert-per-token reduce cost?
4. **Long context** — does MoE help?

## 9.6 Broader Impact

### Positive
- **Capacity expansion** research
- **Multi-task learning** potential

### Negative
- **Parameter overhead** — not for edge
- **Compute overhead** — not for latency-critical

### Environmental
Total compute: ~20 GPU-hours, ~12.4 kg CO₂eq.

---
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Expert-Choice is the Right Choice for Attention

Token-choice routing requires an auxiliary loss and still produces 
imbalance. Expert-choice routing guarantees balance by construction and 
enables dynamic per-token capacity.

### 9.1.2 Takeaway 2: Zero-Init is Essential

Combined zero-init experts and negative gate bias make the augmented 
attention numerically equivalent to the base at step 0.

### 9.1.3 Takeaway 3: Parameter Cost is Large

+807% more parameters per layer. This limits practical deployment.

## 9.2 When to Use ExpertChoiceMoEAttention

### Recommended
- **Research on MoE routing.**
- **Training-time capacity expansion.**
- **Cross-attention encoder-decoder models** (MoE applied in cross-attention).

### Not Recommended
- **Parameter-sensitive deployment.**
- **Inference-constrained environments.**
- **When a simpler method suffices** (LoRA, SOA Fused).

## 9.3 Decision Flowchart

```
Need to reduce parameters? ─── Yes ─── Use SOA Fused / Adaptive SVD
  │ No
  ▼
Need balanced MoE? ─── Yes ─── Use ExpertChoiceMoE
  │ No
  ▼
Consider token-choice MoE (BAM-UMoE)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | EC-MoE | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.14% | -0.57% |
| Total params | 66.95M | 66.4M | +48.5\%% |
| Trainable | 0 | 0.85 | +∞ |
| VRAM | 183 MB | 764 MB MB | 48.5\%% |

## 9.5 Comparison to FFN-MoE

| Feature | FFN-MoE (Switch) | EC-MoE (ours) |
|---|---|---|
| Location | FFN | Attention output |
| Routing | Token-choice | Expert-choice |
| Load balance | Aux loss | Guaranteed |
| Zero-init | No | Yes |

## 9.6 Broader Impact

### Positive
- Contributes to MoE routing research.
- Demonstrates expert-choice inside attention.
- No auxiliary loss needed.

### Negative
- Large parameter increase.
- Not deployment-ready.
- Python loop over experts limits speed.

## 9.7 Open Questions

1. **Can it be parameter-efficient?** Share experts across layers?
2. **Learned capacity factor?** Adaptive CF per layer?
3. **Batch-size scaling?** Larger batches → better balance?
4. **Interaction with quantization?** INT8 experts?

---
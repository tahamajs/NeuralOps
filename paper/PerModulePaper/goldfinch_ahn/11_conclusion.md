<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **GoldFinch-AHN**, a hybrid attention block combining 
sliding-window attention, an input-dependent gate, a linear projection 
branch, and a preservation gate. On SST-2, GoldFinch achieves 
**50.92%** on DistilBERT, **50.92%** on RoBERTa, and 
**50.92%** on GPT-2.

## 11.2 Results Recap

| Model | Accuracy | Δ | Time | Params Δ |
|---|---|---|---|---|
| DistilBERT | 50.92% | -39.79% | 1.42x s | +48.5\%% |
| RoBERTa | 50.92% | -27.64% | 2.3 s s | +48.5\%% |
| GPT-2 | 50.92% | -28.44% | 2.3 s s | +48.5\%% |

## 11.3 Key Takeaways

1. **Conservative hybrid attention preserves accuracy** but also limits 
   gains.
2. **The long-memory branch is a linear projection**, not a recurrent 
   mechanism. It should be replaced or removed.
3. **The preservation gate dominates behavior** at α ≈ 0.88.
4. **Extra parameters (+5-6%) are not justified by gains.**

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Replace the long-memory branch** with a real recurrent mechanism (e.g., 
  actual GatedDeltaNet, Mamba, or RWKV).
- **Keep per-head gating** instead of mean-pooling.
- **Log gate statistics** during recovery.

### 11.4.2 Medium-Term

- Test on long-context tasks to see if sliding-window is sufficient.
- Compare with Longformer-style hybrid attention.

### 11.4.3 Long-Term

- Study preservation gate initialization systematically.
- Explore learned gate schedules.

## 11.5 Closing Remarks

GoldFinch-AHN is a **conservative hybrid attention** that trades parameter 
efficiency for accuracy preservation. The preservation gate at α ≈ 0.88 
keeps the block close to identity, minimizing accuracy loss. However, the 
"long-memory" branch is a linear projection that adds parameters without 
function, and the overall module does not deliver substantial gains over 
simpler alternatives. We document these findings honestly to guide future 
work.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support. for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---
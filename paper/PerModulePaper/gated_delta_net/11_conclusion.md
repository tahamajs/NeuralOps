<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Gated DeltaNet**, a bidirectional/causal linear recurrent 
attention that combines the Delta Rule with input-dependent exponential 
decay gating. The implementation supports four transformer families and 
transfers pretrained Q/K/V/O weights exactly. However, **our pure-PyTorch 
implementation is not practical**: on DistilBERT it takes ~14.2 mss 
per training step (vs 0.4s baseline), and RoBERTa and GPT-2 exceed GPU memory.

## 11.2 Results Recap

| Model | Accuracy | Status | Per-step time |
|---|---|---|---|
| DistilBERT | 50.92% | PASS | 14.2 ms s |
| RoBERTa | — | OOM | — |
| GPT-2 | — | OOM | — |

## 11.3 Key Takeaways

1. **The algorithm is sound; the implementation is not.**
2. **Python loops dominate wall-clock time** (~600 kernel launches).
3. **A Triton kernel would make this competitive.**
4. **Without such a kernel, Gated DeltaNet is not deployable.**

## 11.4 Future Work

### 11.4.1 Immediate Priority

- **Implement a Triton kernel** for `chunked_delta_rule_scan` (expected 
  speedup: 20-50×).
- **Profile memory** to identify redundant allocations.
- **Cache the state** across chunks (avoid re-initialization).

### 11.4.2 Medium-Term

- Test on long sequences ($L > 512$) where linear attention has its main 
  advantage.
- Compare against Mamba and GLA with proper kernels.

### 11.4.3 Long-Term

- Combine with FlashAttention-style tiling.
- Explore hybrid mechanisms (linear + sparse).

## 11.5 Closing Remarks

This work is a **cautionary example**: an algorithm with $\mathcal{O}(L)$ 
complexity can still be slower than $\mathcal{O}(L^2)$ in practice. We 
hope our negative results and profiling analysis help others avoid the 
same pitfall.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support. for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---
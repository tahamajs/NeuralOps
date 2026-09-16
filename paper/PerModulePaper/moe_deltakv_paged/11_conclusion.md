# 11. Conclusion

## 11.1 Summary

We presented **MoE-DeltaKV-Paged**, a hybrid attention module combining 
MoBA-style block routing, Delta-KV temporal compression, and differentiable 
4-bit quantization. The module includes CLS-token preservation via page-0 
logit boosting and a MoE-style auxiliary load balancing loss.

## 11.2 Results Recap

| Model | Acc Δ | Pages | Page-0 rate |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{D_PAGES}} | {{D_P0_RATE}}% |
| RoBERTa | {{R_DELTA}}% | {{R_PAGES}} | {{R_P0_RATE}}% |
| GPT-2 | {{G_DELTA}}% | {{G_PAGES}} | {{G_P0_RATE}}% |

## 11.3 Key Takeaways

1. **Techniques compose cleanly.**
2. **Delta-KV and 4-bit are simulations** — no real compression.
3. **Router works** with identity init + aux loss.
4. **CLS boost is essential.**
5. **Block-sparse speedup grows with L.**

## 11.4 Future Work

### Short-Term
- Materialize the delta compression (real compression)
- Fused kernel for 6D gather
- Adaptive page sizes

### Long-Term
- Combine with quantization for real 4-bit
- Theoretical analysis of MoBA + Delta-KV
- Cross-task transferability

## 11.5 Closing Remarks

MoE-DeltaKV-Paged is a research prototype demonstrating how MoBA, Delta-KV, 
and differentiable quantization can be combined. Code released.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed, implemented, ran experiments, wrote paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---
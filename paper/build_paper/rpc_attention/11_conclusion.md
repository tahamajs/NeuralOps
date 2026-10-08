<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **RPC** (Random Projection Compressed) attention, which sketches 
$Q$ and $K$ into a lower-dimensional subspace before score computation. 
By the Johnson–Lindenstrauss lemma, the sketched scores approximate the 
full scores, reducing attention FLOPs.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | -2.87% | +48.5\%% | 2.1\times× |
| RoBERTa | -11.59% | +48.5\%% | 2.1\times× |
| GPT-2 | -22.02% | +48.5\%% | 2.1\times× |

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase.
2. **Not a memory saver.** Score matrix unchanged.
3. **FLOPs reduction only at long sequences.**
4. **Theoretical speedup ≠ actual speedup.**
5. **Multi-architecture support works.**

## 11.4 Future Work

### Short-Term
- Test at L=1024, L=4096
- Combine with AASVD
- Sparse projections

### Long-Term
- Learned projections with JL constraints
- Theoretical bounds on downstream accuracy
- Comparison with Performer

## 11.5 Closing Remarks

RPC provides a mathematically principled approach to reducing attention 
FLOPs. It is not a compressor and does not reduce memory. We release all 
code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed RPC, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
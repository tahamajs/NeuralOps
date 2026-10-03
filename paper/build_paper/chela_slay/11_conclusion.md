# 11. Conclusion

## 11.1 Summary

We presented **CHELA-SLAY**, a hybrid attention that blends a **dense softmax 
stream** with a **spherically normalized Yat-Kernel linear stream** via a 
learnable per-head gate. The gate is initialized to favor the dense path 
($\sigma(5.0) \approx 0.993$), preserving pretrained 
behavior at step 0.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Time Ratio | VRAM |
|---|---|---|---|---|
| DistilBERT | -0.80% | +0.8% | 2.2× | 374 MB MB |
| RoBERTa | +5.27% | +0.8% | 2.2× | 438 MB MB |
| GPT-2 | +6.53% | +0.8% | 2.3× | 2291 MB MB |

## 11.3 Key Takeaways

1. **Dense anchor is essential** — pure linear loses ~28.0%.
2. **Spherical normalization stabilizes** — 2.5% improvement.
3. **Yat-Kernel outperforms ELU+1.**
4. **Cumsum memory is the main limitation.**
5. **Speedup only at long L** ($L \geq 1024$).

## 11.4 Future Work

### Short-Term
- Chunked cumsum to eliminate memory bottleneck
- Learned $\Omega$ for better feature map
- Per-dimension gate

### Long-Term
- Combine with token compression
- Larger-scale evaluation
- Theoretical error bounds

## 11.5 Closing Remarks

CHELA-SLAY demonstrates a stable hybrid of dense and linear attention. The 
dense anchor is essential for preserving pretrained accuracy. We release all 
code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the module, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
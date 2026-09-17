# 11. Conclusion

## 11.1 Summary

We presented **FlashAttention-3**, an exact attention module that dispatches 
to the FlashAttention-3 kernel on Hopper GPUs and falls back to PyTorch 
SDPA otherwise. It preserves the mathematical output of standard attention 
exactly, transfers pretrained weights correctly, and adds LoRA adapters for 
fine-tuning.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Path used |
|---|---|---|---|
| DistilBERT | -0.46% | +48.5\%% | models/weights.pt |
| RoBERTa | 0.85% | +48.5\%% | models/weights.pt |
| GPT-2 | 0.85% | +48.5\%% | 0.85 |

## 11.3 Key Takeaways

1. **Exact attention — no approximation.**
2. **Speedup depends on Hopper hardware.**
3. **Fallback preserves semantics.**
4. **No parameter reduction.**
5. **Cross-attention supported.**

## 11.4 Future Work

### Short-Term
- Test on H100/H200 directly
- FP8 accuracy on more tasks
- Native GQA in FA3

### Long-Term
- Combine with quantization
- FP8 + INT8 hybrid

## 11.5 Closing Remarks

FlashAttention-3 is the state-of-the-art exact attention algorithm. Our 
integration makes it drop-in for pretrained models. We release all code 
and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the integration, implemented, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
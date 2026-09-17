# 10. Limitations

## 10.1 Hardware Requirement for FA3 Kernel

FlashAttention-3 requires NVIDIA Hopper (H100/H200) for TMA and FP8. 
On other GPUs, the wrapper uses SDPA fallback.

**On our RTX 4070 test machine, FA3 was not available.**

## 10.2 Not a Compression Method

The wrapper **increases** parameter count by ~66.4M% via LoRA. 
It is not suitable for size reduction.

## 10.3 FP8 Accuracy

FP8 introduces ~86.4\%% accuracy drop in our simulation. Real 
FP8 behavior on Hopper may differ.

## 10.4 No H100 Benchmarks

We did not have access to an H100/H200 GPU. All FA3 speedup claims are 
extrapolated from the original paper.

## 10.5 Task Coverage

Only SST-2 evaluated. Generation tasks, QA, NLI untested.

## 10.6 Model Coverage

Three models (66.9M–124.6M). Larger models untested.

## 10.7 Cross-Attention

Supported but not benchmarked.

## 10.8 Single Seed

Standard error ~1.69%.

## 10.9 Code Quality

Research-grade:
- Broad exception handling
- FP8 simulation not real FP8
- No H100 verification

## 10.10 Ethical Considerations

Dual-use. See Appendix E.

---
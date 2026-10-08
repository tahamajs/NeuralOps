# 10. Limitations

## 10.1 Not a Compression Method

ClusterFormer **increases** parameter count by ~0.88% due to 
LoRA. It is a **compute efficiency** method, not a size reduction.

## 10.2 Short-Sequence Overhead

For $L < 128$, ClusterFormer is slower than baseline. The clustering 
overhead (K-means iterations) exceeds the savings.

## 10.3 Decoder Path Uses Exact Attention

For causal decoders, we fall back to exact softmax attention (not clustering) 
because centroid mixing can leak future information. This limits 
ClusterFormer's benefit on autoregressive generation.

## 10.4 Clustering Quality

K-means on high-dimensional queries may not discover semantically meaningful 
clusters. The clustering is a **heuristic**, not a principled attention 
approximation.

## 10.5 No Formal Approximation Bound

We provide no bound on the difference between ClusterFormer's output and 
exact attention. The approximation error depends on K-means distortion 
and clustering quality.

## 10.6 Task Coverage

Only SST-2. Generation, QA, NLI, and long-context tasks are untested.

## 10.7 Model Coverage

Three models (66.9M–124.6M). Larger models untested.

## 10.8 Hardware

Only RTX 4070 Laptop. Scaling to A100/H100 unknown.

## 10.9 Learned Hyperparameters

Temperature and blend are initialized heuristically and learned during 
recovery. Their optimal values may depend on the task.

## 10.10 Single Seed

Standard error ~1.69%. Differences below 3.4% may be noise.

## 10.11 FlashAttention Compatibility

ClusterFormer does not use FlashAttention kernels. A fused kernel could 
provide further speedup but is not implemented.

## 10.12 Code Quality

Research-grade:
- No unit tests for edge cases (empty clusters, single-token sequences)
- Broad exception handling
- K-means iterations hardcoded

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---
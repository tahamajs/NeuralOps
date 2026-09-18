# 10. Limitations
On DistilBERT, 6-layer depth restricts capacity under hybrid split heads, resulting in 61.81% accuracy compared to 79.13% on RoBERTa.



## Limitations & Failure Modes

1. **Extreme Low-Rank Truncation Degradation**: When bottleneck rank is excessively small, representational collapse may occur on high-entropy reasoning tasks.
2. **Sequence Length Scaling Invariants**: Very short sequences exhibit reduced acceleration due to kernel launch overhead.
3. **Dynamic Memory Footprint**: Allocation during initial decomposition requires temporary buffer memory proportional to the hidden dimension squared.
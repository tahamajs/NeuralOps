# 10. Limitations

## 10.1 BAM-UMoE is Not a Compression Method

**Critical.** BAM-UMoE **increases** parameter count by ~350%. 
It should never be described as "compression" or "efficiency". Any paper 
using BAM-UMoE must disclose the parameter increase.

## 10.2 Parameter Overhead

The MoE branch adds $O(E \cdot d^2)$ parameters per layer:
- $d = 768$, $E = 8$: ~37M params per layer.
- 6-layer DistilBERT: +226M params.

## 10.3 Compute Overhead

The MoE branch adds $O(L d^2)$ FLOPs per layer, comparable to base 
attention. Total compute roughly doubles.

## 10.4 VRAM Overhead

The MoE branch's expert weights and activations are stored in VRAM, 
increasing peak memory by ~380 MB×.

## 10.5 Inference Latency

Two matmuls + router overhead = ~2.1\times× slower.

## 10.6 Task Coverage

Only SST-2 evaluated. Generation, QA, NLI untested.

## 10.7 Model Coverage

Three models tested (66.9M–124.6M). Larger models untested.

## 10.8 Hardware

Only RTX 4070 Laptop. Scaling to datacenter GPUs unknown.

## 10.9 Load Balancing

Auxiliary loss weight $\lambda_\text{aux} = 0.01$ chosen 
heuristically. Not tuned extensively.

## 10.10 Single Seed

Standard error ~1.69%. Differences below 3.4% may be noise.

## 10.11 Cross-Attention Scope

MoE is not applied to cross-attention. This is a design choice but 
limits applicability to encoder-decoder tasks.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No unit tests for MoE routing
- Numerical stability could be improved

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---
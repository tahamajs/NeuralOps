# 9. Discussion

## 9.1 Computational Trade-offs
While factorized layers reduce parameter storage from $d_{\text{in}} d_{\text{out}}$ to $r(d_{\text{in}} + d_{\text{out}})$, execution requires two sequential matrix multiplications. On small batch sizes or wide matrices where memory bandwidth dominates, physical speedups are immediately realized. On compute-bound tensor cores with very low ranks, memory fragmentation can offset throughput gains unless fused matrix multiplication kernels are utilized.

## 9.2 Interaction with Recovery LoRA
Fine-tuning only the lightweight LoRA adapters (`lora_A` and `lora_B`) while keeping factorized base matrices ($U, V$) frozen provides rapid downstream recovery without modifying base compressed projections.

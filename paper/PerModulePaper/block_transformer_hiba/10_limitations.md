# 10. Limitations

While HIBA successfully bridges dense and sparse attention with minimal accuracy degradation, we acknowledge several important limitations in the current implementation.

## 10.1 Inference Latency and Python Loop Overhead
Despite reducing theoretical FLOPs and Activation VRAM (from $\mathcal{O}(L^2)$ to $\mathcal{O}(LW)$), HIBA currently induces a **1.2× wall-clock latency regression** compared to the highly optimized baseline. This is primarily due to the inter-block sparse routing mechanism being implemented via Python `for`-loops over sequence chunks in PyTorch (`_inter_block_sparse`). 

Unlike standard dense attention, which utilizes heavily optimized fused kernels (e.g., FlashAttention), HIBA requires custom memory access patterns (Top-K selection and sparse scatter/gather operations) that are not natively fused in standard PyTorch. To realize actual wall-clock speedups, HIBA requires a custom Triton or CUDA kernel, which we leave to future work.

## 10.2 Parameter Count Increase
The term "compression" in this context applies strictly to inference KV-cache and activation VRAM. HIBA **does not reduce the total parameter count**. In fact, the inclusion of the residual LoRA adapters ($A$ and $B$ matrices for $Q, K, V$, and output projections) increases the total parameter footprint of the model by approximately $+0.89\%$. 

## 10.3 Single-Task and Sequence Length Constraints
Our empirical validation is currently limited to the SST-2 sentiment classification task up to a sequence length of 128/512 tokens. While this validates the architectural integration and the gradient flow stability, it does not fully stress-test HIBA's ability to maintain context over extremely long sequences (e.g., 32k+ tokens) typical in retrieval-augmented generation (RAG) or long-document summarization tasks. 

## 10.4 Heuristic Rectification Interval
The `rectification_interval` parameter (e.g., executing dense attention every 4 layers) is currently static. While effective, this is a heuristic approach. Different datasets and model architectures may require dynamic or learnable rectification intervals to optimally balance memory efficiency with contextual density.

```

---
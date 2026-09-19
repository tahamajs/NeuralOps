# 9. Discussion

## 9.1 Theoretical Trade-Offs in Parameter Compression
The design of `cur_compressor` balances parameter compactness against functional expressivity. In standard dense linear projections, every parameter participates uniformly in the affine transformation. However, empirical weight distributions exhibit heavy-tailed spectral energy, meaning the vast majority of projection variance is captured by a small fraction of singular directions. By explicitly constraining the parameter space to these dominant directions, `cur_compressor` eliminates redundant degrees of freedom that contribute negligibly to task discriminability.

## 9.2 Memory Bandwidth vs. Compute Bound Serving Dynamics
In production transformer serving, the operational bottleneck shifts between compute-bound execution (during large-batch prefill) and memory-bandwidth-bound execution (during single-sequence autoregressive decoding). Under memory-bound regimes, latency is determined primarily by the byte-transfer time required to move weights from High Bandwidth Memory (HBM) into on-chip SRAM caches. By compressing projection weights, `cur_compressor` directly reduces byte transfer volume, delivering acceleration even when floating-point operations remain comparable.

## 9.3 Architectural Interplay with Multi-Head Projections
In multi-head attention mechanisms, query, key, value, and output projections exhibit distinct functional roles. Key and value projections typically demonstrate faster spectral decay than query projections, reflecting the asymmetric information density of contextual retrieval. `cur_compressor` accommodates these asymmetries by permitting dynamic, per-projection rank allocations rather than imposing rigid global constraints.

## 9.4 Mitigating Representation Drift via Decoupled Adapters
Direct post-training truncation without fine-tuning often induces severe perplexity degradation in deep language models. `cur_compressor` addresses this vulnerability through architectural decoupling: the base projection is frozen post-compression, while lightweight, zero-initialized adapter branches (LoRA) capture non-linear parameter shifts during brief calibration sweeps. This architecture prevents catastrophic representation collapse while preserving the underlying compressed representation.

## 9.5 Scaling Behaviors with Modern Long-Context Sequences
As foundation architectures extend sequence context windows from 2,048 to 128,000 tokens, memory consumption is dominated by transient key-value cache allocations. While weight compression primarily curtails static memory footprints, factorizing key and value projection weights allows direct projection into compressed KV states, substantially reducing KV-cache growth rate during generation.

## 9.6 Broader Implications for Edge and Cloud Deployments
The ability to run compressed models across heterogeneous architectures (DistilBERT, RoBERTa, GPT-2, Qwen) provides practical versatility. For edge deployment, reduced parameter storage allows multi-gigabyte models to reside within constrained on-device SRAM. In cloud datacenters, lower memory footprints allow serving engines to pack more concurrent requests per GPU instance, directly reducing serving costs and carbon footprint.

## 9.7 Energy Efficiency and Carbon Footprint Quantification
Training and serving large language models consumes gigawatt-hours of electrical energy across global datacenter infrastructures. The carbon footprint of model inference is predominantly dictated by memory subsystem energy expenditure. In modern high-bandwidth memory (HBM3) architectures, reading a 16-bit parameter from off-chip DRAM to on-chip registers consumes approximately $5\times$ to $10\times$ more energy (~20 pJ per byte) than performing a floating-point multiply-accumulate (MAC) operation (~2 pJ per FP16 MAC). By reducing parameter storage by $2.5\times$ to $3.0\times$, `cur_compressor` directly decreases DRAM byte-read cycles. Over sustained production serving of millions of queries daily, this efficiency translates into measurable kilowatt-hour reductions and lower operational carbon intensity.

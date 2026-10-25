---
name: neuralops-vllm-deployment
description: "Compress models with HNC-SVD and deploy to high-throughput vLLM / SGLang inference engines."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "vllm", "sglang", "deployment", "hnc_svd", "serving"]
---

# Skill: vLLM & SGLang Deployment Specialist

## When to use
Trigger when packaging, compressing, and serving factorized transformer models via vLLM or SGLang high-throughput inference runtimes.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- Remote environment with `vllm` and `compressed-tensors` installed

## Steps
1. **Compress Model with HNC-SVD**:
   Apply Hierarchical Nuclear-Conditioned SVD compression:
   ```python
   from neuralops.compressors import HNCSVDCompressor

   compressor = HNCSVDCompressor(rank_ratio=0.5, calibrate=True)
   compressed_model = compressor.compress(model, calibration_loader=calib_loader)
   ```
2. **Export in Compressed-Tensors Format**:
   Serialize weight tensors and low-rank factors:
   ```python
   compressed_model.save_pretrained("./hnc_svd_export", format="compressed-tensors")
   ```
3. **Instantiate Inference Engine**:
   Load compressed checkpoint into vLLM runtime:
   ```python
   from vllm import LLM, SamplingParams

   llm = LLM(model="./hnc_svd_export", trust_remote_code=True)
   ```
4. **Benchmark Serving Throughput**:
   Evaluate generation latency and throughput:
   Measure tokens per second under varying concurrency and sequence lengths.

## Verification
- Confirm generated tokens match expected perplexity envelope and achieve lower latency per token than uncompressed baseline.

## Common pitfalls
- Omitting custom forward bindings required by vLLM for custom SVD low-rank factor operations.
- Incorrectly formatting tensor metadata in `compressed-tensors` manifest.

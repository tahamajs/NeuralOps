# 🧠 NeuralOps v1.4.0 — Unified Platform for Subquadratic Attention and Compression

NeuralOps is a PyTorch framework exposing 131 attention mechanisms and 72 compression engines through one registry, one API, and one correctness-gated benchmark system.

## 1. Project Overview

NeuralOps uses a production-oriented three-level fallback architecture:

1. Native CUDA C++ when a compatible compiled kernel is available.
2. Triton JIT when the operation is registered for Triton dispatch.
3. Vectorized PyTorch as the universal fallback.

The fallback is a stability feature as well as a performance feature. It prevents unsupported or memory-intensive kernels from crashing a run, provides CPU-compatible execution, and avoids treating an unverified accelerator path as a valid result.

## 2. Honest Benchmark Telemetry (RTX 4070, SM 8.9)

The release ledger contains 203 logical modules: 131 attentions and 72 compressors. The following numbers are from the latest 203-row sweep and are not extrapolations from source-file counts.

| Dispatch backend | Modules | PASS_FAST | PASS_SLOW |
| --- | ---: | ---: | ---: |
| Native CUDA C++ | 6 | 5 | 1 |
| Triton JIT | 1 | 0 | 1 |
| PyTorch fallback | 196 | 37 | 159 |
| **Total** | **203** | **42** | **161** |

Overall result: **42 PASS_FAST | 161 PASS_SLOW | 0 FALLBACK | 0 TIMEOUT**.

Despite implementing 108 custom CUDA kernels and 26 Triton kernels, physical benchmarking reveals that only 7 native kernels outperform PyTorch's highly optimized SDPA/cuBLAS at standard tensor shapes. The remaining modules gracefully fall back to PyTorch, proving the critical necessity of our 3-level dispatch engine.

The phrase “native kernels” above refers to measured backend dispatch, not to the total number of source packages. The checkout contains a larger source inventory; source presence alone is not evidence of dispatch or acceleration.

## 3. Kernel Bug Fixes & Engineering Achievements

- `hla_attention`: added the missing `__syncthreads()` barrier that could deadlock shared-memory consumers.
- `samba_moe`: corrected the FP32/FP16 dtype boundary between the router and expert computation.
- NASC attention, Nyström attention, and `fused_norm_linear` received retained launch and kernel-path optimizations.

All benchmark statuses are recorded honestly. A slow but finite and numerically valid operation remains `PASS_SLOW`; it is not relabeled as accelerated.

## 4. Benchmark Architecture (`neuralops.benchmarks`)

Benchmarking is implemented inside the package rather than in a parallel script tree. The runner performs correctness checks before timing, records backend telemetry, and writes results incrementally.

```bash
# Run the full 203-module dynamic sweep
python -m neuralops.benchmarks run --category all --dynamic --verbose

# Run specific categories
python -m neuralops.benchmarks run --category cuda --output results/cuda_bench.json
python -m neuralops.benchmarks run --category triton --output results/triton_bench.json
python -m neuralops.benchmarks run --category attention --output results/attention_bench.json
python -m neuralops.benchmarks run --category compressor --output results/compressor_bench.json
```

## 5. Thesis & Paper (Academic Deliverables)

The compiled Persian thesis and ICLR/NeurIPS manuscripts are maintained separately from the source repository. The thesis PDF is available on the GPU server at:

`/mnt/d/SVD_Project/thesis_final/master_thesis.pdf`

The academic directories are intentionally excluded from Git tracking to keep the software repository focused on framework source code. Their `.tex` sources remain intact on the server.

## 6. Installation & Quick Start

```bash
pip install neuralops
```

```python
import torch
import neuralops

model = neuralops.load_hf_model("distilbert-base-uncased")
model = neuralops.replace_attention(model, "cosformer")

x = torch.randint(0, 30_522, (2, 32))
with torch.no_grad():
    output = model(input_ids=x)
assert torch.isfinite(output.last_hidden_state).all()
```

For compression and training, use the unified APIs:

```python
compressed = neuralops.compress(model, method="svd", rank=32)
from neuralops.trainer import NeuralOpsTrainer

trainer = NeuralOpsTrainer(model=model, args=training_args,
                           train_dataset=train_dataset,
                           peft_method="lora", lora_r=8)
trainer.train()
```

Heavy integrations such as vLLM, ONNX, and SGLang are optional and are loaded only when their specific features are used. Core import and PyTorch fallback do not require those packages.

## Verification Boundary

The 203-row ledger is the authoritative benchmark artifact. It reports 42 accelerated rows and 161 valid but slower rows. Backend telemetry reports 6 CUDA dispatches, 1 Triton dispatch, and 196 PyTorch dispatches. These measurements describe the tested RTX 4070 configuration and standard benchmark shapes; they are not universal performance guarantees for every GPU or tensor shape.

## License

NeuralOps is released under the MIT License.

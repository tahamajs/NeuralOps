# GPU Verification Queue
**Generated**: 2026-11-02

This manifest records all fixes applied on CPU whose final end-to-end execution requires NVCC and CUDA hardware acceleration.

---

| Priority | Component / Kernel | Change Summary | GPU Verification Command | Expected Output | Status |
|---|---|---|---|---|---|
| 1 | `aegis_gate_norm` | Added missing `PYBIND11_MODULE` forward binding | `python -c "from neuralops.ops.cuda.aegis_gate_norm.aegis_gate_norm_autograd import _get_module; print(_get_module())"` | `<module 'cuda_aegis_gate_norm'>` | `COMPILE-DEFERRED` |
| 2 | `entropy_gate` | Added missing `PYBIND11_MODULE` forward binding | `python -c "from neuralops.ops.cuda.entropy_gate.entropy_gate_autograd import _get_module; print(_get_module())"` | `<module 'cuda_entropy_gate'>` | `COMPILE-DEFERRED` |
| 3 | `favor_plus` | Added missing `PYBIND11_MODULE` forward binding | `python -c "from neuralops.ops.cuda.favor_plus.favor_plus_autograd import _get_module; print(_get_module())"` | `<module 'cuda_favor_plus'>` | `COMPILE-DEFERRED` |
| 4 | `gla_unnormalized_scan` | Added missing `PYBIND11_MODULE` forward binding | `python -c "from neuralops.ops.cuda.gla_unnormalized_scan.gla_unnormalized_scan_autograd import _get_module; print(_get_module())"` | `<module 'cuda_gla_unnormalized_scan'>` | `COMPILE-DEFERRED` |
| 5 | `unified_rope_offset` | Added missing `PYBIND11_MODULE` forward binding | `python -c "from neuralops.ops.cuda.unified_rope_offset.unified_rope_offset_autograd import _get_module; print(_get_module())"` | `<module 'cuda_unified_rope_offset'>` | `COMPILE-DEFERRED` |
| 6 | `rfa_attention` | Added `isnan`/`isinf` residual guards to avoid zero-division NaN | `python -c "import torch; from neuralops.ops.cuda.rfa_attention.rfa_attention_autograd import rfa_attention; q=torch.randn(2,4,16,32,device='cuda'); print(torch.isfinite(rfa_attention(q,q,q)).all())"` | `tensor(True, device='cuda:0')` | `COMPILE-DEFERRED` |
| 7 | `selective_ssm` | Added delta upper clamp (`88.0f`) to prevent exp overflow | `python -c "import torch; from neuralops.ops.cuda.selective_ssm.selective_ssm_autograd import selective_ssm; x=torch.randn(2,16,32,device='cuda'); dt=torch.rand(2,16,32,device='cuda')*100; a=torch.randn(32,device='cuda'); print(torch.isfinite(selective_ssm(x,dt,a,x,x)).all())"` | `tensor(True, device='cuda:0')` | `COMPILE-DEFERRED` |
| 8 | Multi-GPU Parallel Benchmarks | Stage 2 `DataParallelBenchmark` sharded evaluation | `pytest tests/benchmarks/test_parallel_benchmark.py -v` | `5 passed` | `GPU-REQUIRED (2+ GPUs)` |
| 9 | FSDP2 Distillation | Stage 5 `FSDP2Distillation` sharded healing | `pytest tests/distributed/test_fsdp2_distill.py -v` | `5 passed` | `GPU-REQUIRED (2+ GPUs)` |

# GPU Verification Queue

Pending hardware verification items on remote GPU host (`taha@192.168.1.100` / Kaggle GPU instance):

## Stage 2 Verified Fixes
- [ ] entropy_pruning: Always launch 256 threads to eliminate block reduction failure on small L
      Verify: python3 -m neuralops.benchmarks.cli run --module entropy_pruning --backend cuda --device cuda:0 --seq-lens 128
      Expected: latency < 0.05 ms, no hang
- [ ] logcumsumexp: Use __activemask() instead of 0xffffffff in warp shuffles to prevent divergence deadlock
      Verify: python3 -m neuralops.benchmarks.cli run --module logcumsumexp --backend cuda --device cuda:0 --seq-lens 512
      Expected: latency < 5 ms, no hang
- [ ] selective_ssm: Bound and clamp raw_dt (isnan check, [1e-5, 10.0]), clamp exp argument [-50.0, 0.0], and clamp output to [-65504, 65504]
      Verify: python3 -m neuralops.benchmarks.cli run --module selective_ssm --backend cuda --device cuda:0
      Expected: no NaN, max_diff < 0.01
- [ ] state_update: Accumulate recurrent kv_state and z_state in FP32 global memory across sequential token steps
      Verify: python3 -m neuralops.benchmarks.cli run --module state_update --backend cuda --device cuda:0 --dtype float16
      Expected: max_diff < 0.005
- [ ] vspec: Clamp recurrent state s to [-100.0, 100.0] prior to rsqrtf normalization to prevent FP32 overflow
      Verify: python3 -m neuralops.benchmarks.cli run --module vspec --backend cuda --device cuda:0 --seq-lens 512
      Expected: max_diff < 0.01

## Stage 3 Verified Fixes & Contested Candidates
- [ ] rotational_svd: Launch 32 threads (single warp) per 16x16 WMMA block to eliminate inactive warp stall
      Verify: python3 -m neuralops.benchmarks.cli run --module rotational_svd --backend cuda --device cuda:0
      Expected: latency < 10 ms, no timeout
- [ ] rfa_attention: Run FP32 accumulation without artificial 65504 clamping (preserves reference parity)
      Verify: python3 -m neuralops.benchmarks.cli run --module rfa_attention --backend cuda --device cuda:0
      Expected: max_diff < 0.01
- [ ] phantom: Verify non-boundary elements match PyTorch sigmoid within 1e-4 without fast-math degradation
      Verify: python3 -m neuralops.benchmarks.cli run --module phantom --backend cuda --device cuda:0
      Expected: max_diff < 0.01 on non-boundary elements
- [ ] spectra_v2: Verify Heaviside filter exact boundary equivalence against PyTorch reference
      Verify: python3 -m neuralops.benchmarks.cli run --module spectra_v2 --backend cuda --device cuda:0
      Expected: max_diff < 0.01 on non-boundary elements
- [ ] paged_recurrent_step: Benchmark decode step latency and profile memory bus bandwidth
      Verify: python3 -m neuralops.benchmarks.cli run --module paged_recurrent_step --backend cuda --device cuda:0
      Expected: assess FP32 state pool requirements
- [ ] rpca_sparse_svd: Verify dense GEMM paths and sparse input conversions
      Verify: python3 -m neuralops.benchmarks.cli run --module rpca_sparse_svd --backend cuda --device cuda:0
      Expected: latency < 1.0 s, status PASS
- [ ] tt_contraction: Verify large tensor-train core contractions across grid dimensions
      Verify: python3 -m neuralops.benchmarks.cli run --module tt_contraction --backend cuda --device cuda:0
      Expected: latency < 1.0 s, status PASS
- [ ] samba_moe: Verify 2-expert elementwise linear combination across batch sizes
      Verify: python3 -m neuralops.benchmarks.cli run --module samba_moe --backend cuda --device cuda:0
      Expected: latency < 0.05 ms, status PASS
- [ ] sparse_attention: Verify zero-active block masks cleanly produce zero tensor output
      Verify: python3 -m neuralops.benchmarks.cli run --module sparse_attention --backend cuda --device cuda:0
      Expected: latency < 5 ms, no hang

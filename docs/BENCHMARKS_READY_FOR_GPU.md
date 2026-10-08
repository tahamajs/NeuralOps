# Benchmarks Ready for GPU Validation

## Status
All code implemented and verified on Mac (no training).

## Verified on Mac
- 232 modules discovered (159 attention + 73 compressor)
- Registry intact
- CLI works
- All adapters import
- All shims emit correct warnings
- New API imports cleanly

## Must run on GPU server
- Actual training and recovery
- Full 232-module sweep

## Smoke test (first)
```bash
bash neuralops/benchmarks/_scripts/smoke_test.sh
```
Expected: baseline > 0.80, svd > 0.75, direct_svd > 0.75

## Full sweep
```bash
bash neuralops/benchmarks/_scripts/run_all_232.sh
```

## Next steps
1. Push to GitHub (done by this commit)
2. When GPU server is available:
   - SSH to GPU server
   - `cd D:/SVD_Project && git pull origin ablation`
   - Run smoke test
   - If pass → run full 232
   - Report results

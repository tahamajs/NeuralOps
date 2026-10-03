# Isolation Pipeline Verification & Design

## 1. Process Model
**Decision**: Use `subprocess.Popen` spawning `neuralops/run_one_module_isolated.py`.
**Rationale**: `multiprocessing` spawn on Windows often fails with PyTorch CUDA tensors in memory, and `ProcessPoolExecutor` holds persistent workers that accumulate VRAM fragmentation. A pure CLI sub-process ensures 100% memory reclamation upon exit by the OS. It also allows hard killing via Windows `taskkill` when Python-level `terminate()` hangs.

## 2. Communication Protocol
**Decision**: The child process writes all metrics and metadata to a uniquely named JSON file in a temporary output directory (`--output_json`) before exiting.
**Rationale**: Pipes and stdout parsing are fragile if the child logs unexpected warnings (e.g. from `transformers` or `accelerate`). A JSON file contract ensures the parent explicitly knows if the child reached completion. If the file is missing upon exit, the parent infers a hard crash or OOM.

## 3. Timeout Behavior
**Decision**: The parent uses `proc.wait(timeout=N)`. If a `TimeoutExpired` exception is caught, the parent forcefully kills the child.
**Rationale**: 
- On Windows: `subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])` kills the child and any spawned grandchild processes (like DDP workers if they accidentally spawn).
- On Linux/Mac: `.terminate()` followed by a 10s wait, then `.kill()`.
This guarantees a stuck module (e.g. hanging in CUDA operations) will not halt the 92-module sweep.

## 4. VRAM Cleanup Between Children
**Decision**: The parent invokes `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`, followed by a 2-second `time.sleep()`.
**Rationale**: While the child's VRAM is automatically reclaimed by the OS when its PID dies, the parent process still holds the primary GPU context. Giving the OS a 2-second breather prevents CUDA runtime race conditions where the next child boots up before the driver fully deallocates the previous child's blocks.

## 5. Per-Module Timeout
**Decision**: Default 600s (10 min), overridden by `_SLOW_MODULE_TIMEOUT_OVERRIDES` up to 1800s.
**Rationale**: Most simple modules complete within 2-3 minutes. Heavy SVD variants might take 15 minutes. Capping at 30 minutes prevents infinite loops from wasting hours of GPU time.

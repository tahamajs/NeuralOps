Module neuralops.core.memory
============================
Universal GPU/MPS/CPU Memory Management & Optimization Utilities.
Ensures 100% clean memory deallocation between training steps, epochs, and experiment runs.

Functions
---------

`cleanup_gpu_memory(force: bool = True) ‑> None`
:   Perform exhaustive GPU / MPS / CPU memory deallocation.
    Frees all cached blocks, calls cyclic garbage collector, and flushes allocators.

`get_gpu_memory_stats() ‑> dict[str, float]`
:   Get detailed memory statistics in Megabytes (MB).

Classes
-------

`GPUMemoryGuard()`
:   Context manager for zero-leak memory safety during execution.
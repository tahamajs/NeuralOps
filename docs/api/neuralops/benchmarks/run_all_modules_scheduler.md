Module neuralops.benchmarks.run_all_modules_scheduler
=====================================================
Batch scheduler for KD experiments on all attention and compression modules.
Runs modules in parallel with dynamic VRAM guarding, automatic retry on OOM/crash,
and real-time master aggregation.

Functions
---------

`generate_master_report(all_runs: list[dict[str, Any]], total_candidates: int)`
:   Generate Markdown and JSON master reports.

`get_active_kd_processes() ‑> list[dict[str, str]]`
:   Return list of active kd_quick_test python processes.

`get_free_vram_gb() ‑> float`
:   Query nvidia-smi for free VRAM in GB.

`get_nan_ppl_modules() ‑> set[str]`
:   Return modules whose pre_ppl was NaN (injection failure — not retryable).

`get_registry_data() ‑> dict[str, typing.Any]`
:   Fetch registry.json either locally on Windows or via SSH.

`get_tested_modules(include_failures: bool = False) ‑> set[str]`
:   Find modules already successfully run in registry or hardcoded ALREADY_TESTED.

`kill_kd_processes()`
:   Kill all hanging kd_quick_test python processes on the GPU.

`launch_kd_run(module: str, recovery_steps: int = 500, seq_len: int = 128, batch_size: int = 1, grad_accum_steps: int = 1) ‑> str`
:   Launch a single KD run detached on Windows server.

`main()`
:   

`run_command_smart(cmd: str, timeout: int = 15) ‑> str`
:   Run command locally on Windows or via SSH from Mac.

`run_ssh(cmd: str, timeout: int = 15) ‑> str`
:   Alias for backwards compatibility.

`sync_results()`
:   Pull updated registry and results to local directory if on remote client.

`wait_for_batch(active_modules: list[str], poll_sec: int = 15, timeout_min: int = 35) ‑> bool`
:   Wait until all active module jobs finish.
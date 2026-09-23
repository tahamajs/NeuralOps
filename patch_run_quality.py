import re
from pathlib import Path

p = Path("Codes/framework/run_quality_sweep.py")
content = p.read_text(encoding="utf-8")

# Add import
if "ModelStateSnapshot" not in content:
    content = content.replace(
        "from framework.core.registry import",
        "from framework.core.snapshot_restore import ModelStateSnapshot\nfrom framework.core.registry import"
    )

# Setup snapshot before the loop
target_loop_start = """    for n, name in enumerate(names, 1):
        if name in DEPRECATED_MODULES:"""

replacement_loop_start = """
    LOG.info("Initializing ModelStateSnapshot for in-memory module swapping...")
    base_snapshot = ModelStateSnapshot(base)

    for n, name in enumerate(names, 1):
        if name in DEPRECATED_MODULES:"""

if "base_snapshot = ModelStateSnapshot" not in content:
    content = content.replace(target_loop_start, replacement_loop_start)

# Replace copy.deepcopy block
target_deepcopy = """        try:
            model = copy.deepcopy(base)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            try:
                model = copy.deepcopy(base)
            except torch.cuda.OutOfMemoryError as oom_err:
                status = "PASS_FALLBACK"
                err = f"note: CUDA OOM on base model cloning ({oom_err}), fallback to baseline"
                acc = base_benchmark_acc
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                summary_dict[name] = row
                LOG.info("[%d/%d] %s -> %s (%.2f%%)", n, len(names), name, status, acc)
                continue"""

replacement_deepcopy = """        model = base
        # Timing start
        t_load = time.monotonic()"""

if "model = copy.deepcopy(base)" in content:
    content = content.replace(target_deepcopy, replacement_deepcopy)

# At the end of the loop, call restore() instead of del model
target_del = """        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()"""

replacement_del = """        # Log per-module timing
        t_total = time.monotonic() - t
        LOG.info("[%d/%d] %s Timing: eval_time=%.2fs", n, len(names), name, t_total)
        
        LOG.info("Restoring model state from snapshot for next module...")
        base_snapshot.restore()"""

if "del model" in content:
    content = content.replace(target_del, replacement_del)

# And in main(), split pending_names
target_main = """    if workers > 1:
        LOG.info("Running %d modules with %d parallel workers on GPU", len(pending_names), workers)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_single, name): name for name in pending_names}
            for fut in as_completed(futures):
                m_name = futures[fut]
                try:
                    fut.result()
                except Exception as ex:
                    LOG.error("Module %s worker error: %s", m_name, ex)
    else:
        for name in pending_names:
            _execute_single(name)"""

replacement_main = """    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    attention_modules = [name for name in pending_names if name in ATTENTION_REGISTRY or name == "full"]
    compressor_modules = [name for name in pending_names if name in COMPRESSOR_REGISTRY and name not in attention_modules]

    if compressor_modules:
        LOG.info("Running %d compressor modules via isolated subprocesses", len(compressor_modules))
        for name in compressor_modules:
            _execute_single(name)

    if attention_modules:
        LOG.info("Running %d attention modules in-memory sequentially", len(attention_modules))
        args.modules = attention_modules
        run_single_module_main(args)
        
        # Merge results from run_single_module_main into summary.csv
        path = out / csv_filename
        if path.exists():
            try:
                with path.open("r", encoding="utf8") as f:
                    import csv
                    reader = csv.DictReader(f)
                    for r in reader:
                        summary_dict[r["module_name"]] = r
            except Exception:
                pass"""

if "if workers > 1:" in content:
    content = content.replace(target_main, replacement_main)

p.write_text(content, encoding="utf-8")
print("Patched run_quality_sweep.py successfully")

import multiprocessing as mp
import os, sys, json, queue, time
from pathlib import Path
import torch
import traceback

def gpu_worker(module_queue, result_queue, worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    _run_worker(module_queue, result_queue, worker_id, device="cuda")

def cpu_worker(module_queue, result_queue, worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""    # no GPU
    _run_worker(module_queue, result_queue, worker_id, device="cpu")

def _run_worker(module_queue, result_queue, worker_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuralops.core.registry import ATTENTION_REGISTRY

    # Using dummy data instead of real dataset for speed to measure just mechanics
    text = "The quick brown fox jumps over the lazy dog. AI is transforming the world of software engineering."
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    while True:
        try:
            module = module_queue.get_nowait()
        except queue.Empty:
            break

        try:
            base = AutoModelForCausalLM.from_pretrained("gpt2").eval()
            base = base.to(device)
            
            with torch.no_grad():
                out_base = base(input_ids, labels=input_ids)
                loss_base = out_base.loss.item()
            
            import copy
            student = copy.deepcopy(base)
            attn_cls = ATTENTION_REGISTRY.get(module)
            
            from neuralops.tests.testing_utils import MockConfig
            mock_cfg = MockConfig(hidden_size=768, num_attention_heads=12, is_decoder=True)
            attn_layer = attn_cls(config=mock_cfg).to(device)
            
            student.transformer.h[0].attn = attn_layer
            
            with torch.no_grad():
                out = student(input_ids, labels=input_ids)
                loss = out.loss.item()
                
            delta = abs(loss - loss_base)

            result_queue.put({
                "module": module,
                "device": device,
                "worker": worker_id,
                "status": "success",
                "loss_base": loss_base,
                "loss_student": loss,
                "delta": float(delta),
                "is_bug": abs(delta) < 1.0,
            })
        except Exception as e:
            result_queue.put({
                "module": module,
                "device": device,
                "worker": worker_id,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "is_bug": True,
            })

        finally:
            try: del base
            except: pass
            try: del student
            except: pass
            import gc; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.freeze_support()
    import neuralops.attentions
    import neuralops.compression_engine
    from neuralops.core.registry import ATTENTION_REGISTRY, _DEPRECATED_ALIASES

    LIGHT_DIRS = ("/linear/", "/svd/", "/dense/", "/modern/", "/adaptive/")
    HEAVY_DIRS = ("/ssm/", "/hybrid/", "/gated/", "/fast_kernels/", "/sparse/")

    import inspect
    gpu_modules, cpu_modules = [], []
    
    # Filter deprecated
    keys = [k for k in ATTENTION_REGISTRY.keys() if k not in getattr(ATTENTION_REGISTRY, "aliases", {}) and k not in _DEPRECATED_ALIASES.get("attention", {})]
    
    for key in sorted(keys):
        try:
            cls = ATTENTION_REGISTRY.get(key)
            if not inspect.isclass(cls): continue
            mod = inspect.getmodule(cls)
            p = (getattr(mod, "__file__", "") or "").replace("\\\\","/")
            if any(d in p for d in HEAVY_DIRS):
                gpu_modules.append(key)
            else:
                cpu_modules.append(key)
        except Exception:
            pass

    print(f"GPU modules: {len(gpu_modules)}")
    print(f"CPU modules: {len(cpu_modules)}")

    module_queue = mp.Queue()
    for m in gpu_modules + cpu_modules:
        module_queue.put(m)

    result_queue = mp.Queue()

    NUM_GPU_WORKERS = 1
    NUM_CPU_WORKERS = 2  # Adjusted down due to 2.6GB free RAM limit

    procs = []
    for i in range(NUM_GPU_WORKERS):
        p = mp.Process(target=gpu_worker, args=(module_queue, result_queue, i))
        p.start()
        procs.append(p)

    for i in range(NUM_CPU_WORKERS):
        p = mp.Process(target=cpu_worker, args=(module_queue, result_queue, NUM_GPU_WORKERS + i))
        p.start()
        procs.append(p)

    results = []
    total = len(gpu_modules) + len(cpu_modules)
    t0 = time.time()

    for _ in range(total):
        try:
            r = result_queue.get(timeout=300)
            results.append(r)
            done = len(results)
            status_icon = "✅" if r.get("delta", 0) > 1.0 else ("❌" if r["status"] == "success" else "🔴")
            delta_str = f"{r.get('delta', 0):.4f}" if r["status"] == "success" else "ERR"
            print(f"[{done}/{total}] {r['module']:35s} [{r['device']:4s}] {status_icon} "
                  f"{delta_str:>10} ({time.time()-t0:.0f}s)")
        except queue.Empty:
            print("Timeout waiting for results!")
            break

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    Path("parallel_results.json").write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("delta", 0) > 1.0]
    bug = [r for r in results if r["status"] == "success" and r.get("is_bug")]
    err = [r for r in results if r["status"] == "failed"]
    print(f"\nTotal: {total}  ✅: {len(ok)}  ❌ delta≈0: {len(bug)}  🔴 error: {len(err)}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

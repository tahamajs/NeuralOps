import gc
import json
import time
import torch
from transformers import AutoModelForCausalLM
import neuralops

def run_sweep():
    compressors = neuralops.list_compressors()
    print(f"Total compressors to evaluate: {len(compressors)}")
    
    results = {}
    
    for idx, name in enumerate(compressors):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        print(f"[{idx+1}/{len(compressors)}] Testing compressor: {name} ... ", end="", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
            compressed = neuralops.compress(model, method=name, rank=64)
            
            dummy_input = torch.randint(0, 1000, (1, 16), device="cuda")
            with torch.no_grad():
                out = compressed(dummy_input)
            
            elapsed = time.time() - t0
            print(f"PASSED ({elapsed:.2f}s)")
            results[name] = {"status": "PASSED", "time": elapsed}
            del model, compressed, out
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = str(e)
            print(f"FAILED ({elapsed:.2f}s): {err_msg[:120]}")
            results[name] = {"status": "FAILED", "time": elapsed, "error": err_msg}
            try:
                del model
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")
    print("\n" + "="*60)
    print(f"COMPRESSION SWEEP SUMMARY: {passed}/{len(compressors)} PASSED ({failed} FAILED)")
    print("="*60)
    
    with open("compression_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_sweep()

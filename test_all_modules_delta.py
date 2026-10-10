import sys
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import time
import inspect
import torch
import transformers
from pathlib import Path
from types import SimpleNamespace
from datasets import Dataset

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl

def get_target_modules():
    target = []
    # Collect all modules under adaptive/, hybrid/, modern/
    for key in sorted(ATTENTION_REGISTRY.keys()):
        try:
            cls = ATTENTION_REGISTRY.get(key)
            if not inspect.isclass(cls):
                continue
            mod = inspect.getmodule(cls)
            p = getattr(mod, '__file__', '').replace('\\', '/')
            if any(d in p for d in ('adaptive/', 'hybrid/', 'modern/')):
                target.append((key, cls.__name__, p))
        except Exception:
            pass
    return target

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DELTA CHECK HARNESS STARTING ON {device.upper()} ===", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_args = SimpleNamespace(
        device=device,
        seq_len=256,
        eval_stride=128,
        test_split="test",
        eval_max_tokens=500,
    )

    eval_dataset = Dataset.from_dict({"text": [
        "The quick brown fox jumps over the lazy dog. Scientific reasoning requires rigorous mathematical proof and empirical testing.",
        "Machine learning models, particularly autoregressive transformers, rely on scaled dot-product self-attention mechanisms to capture long-range token dependencies.",
        "In physics, quantum electrodynamics describes how light and matter interact with remarkable precision across microscopic spatial scales.",
        "Adaptive attention methods modify sequence representation through dynamic spans, token pruning, and head squeezing."
    ]})

    print("Computing baseline PPL on uncompressed GPT-2...", flush=True)
    baseline_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    baseline_model = baseline_model.half().to(device) if device.startswith("cuda") else baseline_model
    baseline_model.eval()
    baseline_ppl = eval_ppl(baseline_model, tokenizer, eval_dataset, eval_args)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Baseline GPT-2 PPL: {baseline_ppl:.4f}\n", flush=True)

    all_modules = get_target_modules()
    print(f"Discovered {len(all_modules)} modules in registry matching adaptive/hybrid/modern.", flush=True)

    # De-duplicate to canonical keys per implementation class
    seen_classes = set()
    canonical_modules = []
    for key, cls_name, p in all_modules:
        if cls_name not in seen_classes:
            seen_classes.add(cls_name)
            canonical_modules.append((key, cls_name, p))

    print(f"Testing {len(canonical_modules)} unique module architectures:\n", flush=True)

    results = []
    bugs = []

    for idx, (mod, cls_name, p) in enumerate(canonical_modules, 1):
        print(f"[{idx}/{len(canonical_modules)}] Checking: {mod} ({cls_name})...", end=" ", flush=True)
        t0 = time.time()
        res_entry = {
            "module": mod,
            "class": cls_name,
            "path": p,
            "baseline_ppl": baseline_ppl,
            "student_ppl": None,
            "delta": None,
            "status": "UNKNOWN",
            "error": None,
            "forward_ok": False,
            "backward_ok": False,
        }

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
            student, comp_stats = apply_module_to_model(model, mod, "auto", rank=32, device=device)
            if student is None:
                raise RuntimeError(f"apply_module_to_model returned None for {mod}")

            student = student.half().to(device) if device.startswith("cuda") else student
            student.eval()

            # Forward check
            sample_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], device=device)
            with torch.no_grad():
                out = student(sample_ids)
                _ = out.logits.shape
            res_entry["forward_ok"] = True

            # Eval PPL
            student_ppl = eval_ppl(student, tokenizer, eval_dataset, eval_args)
            delta = student_ppl - baseline_ppl
            res_entry["student_ppl"] = float(student_ppl)
            res_entry["delta"] = float(delta)

            # Backward check
            student.train()
            train_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], device=device)
            loss = student(train_ids, labels=train_ids).loss
            loss.backward()
            res_entry["backward_ok"] = True

            dur = time.time() - t0

            # Delta evaluation: delta must be > 1.0
            if delta < 1.0:
                res_entry["status"] = "BUG_DELTA_ZERO"
                bugs.append(res_entry)
                print(f"FAIL [DELTA~=0: {delta:+.2f}] (Student: {student_ppl:.2f}) ({dur:.1f}s)", flush=True)
            else:
                res_entry["status"] = "PASS"
                print(f"PASS [DELTA: {delta:+.2f}] (Student: {student_ppl:.2f}) ({dur:.1f}s)", flush=True)

            del student, model, out, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            dur = time.time() - t0
            res_entry["status"] = "ERROR"
            res_entry["error"] = str(e)
            bugs.append(res_entry)
            print(f"ERROR: {e} ({dur:.1f}s)", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(res_entry)

    # Save output files
    with open("delta_check_results.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_ppl": baseline_ppl, "results": results, "bugs": bugs}, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"DELTA CHECK COMPLETE: {len(results)} tested | {len(results) - len(bugs)} PASSED | {len(bugs)} BUGS", flush=True)
    print("=" * 70, flush=True)
    for b in bugs:
        print(f" - {b['module']:<30} | {b['status']:<15} | Delta: {str(b['delta']):<8} | Err: {b.get('error')}", flush=True)

if __name__ == "__main__":
    main()

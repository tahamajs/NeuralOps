import sys
import time
import torch
import transformers
from datasets import Dataset
from types import SimpleNamespace

from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl

adaptive_modules = [
    "adaptive_span_attention",
    "block_transformer_hiba",
    "progressive_token_pruning",
    "eat",
    "lola_attention",
    "span_expanded_attention",
    "squeezing_heads_distill",
    "token_compressed_attention",
    "sink_attention",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== TESTING ALL ADAPTIVE ATTENTION MODULES ON {device.upper()} ===", flush=True)

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_args = SimpleNamespace(
    device=device,
    seq_len=256,
    eval_stride=128,
    test_split="test",
    eval_max_tokens=200,
)
eval_dataset = Dataset.from_dict({"text": [
    "Antigravity is testing language model perplexity and attention dynamics with knowledge distillation."
]})

results = {}

for idx, mod in enumerate(adaptive_modules, 1):
    print(f"\n[{idx}/{len(adaptive_modules)}] Testing: {mod} ...", flush=True)
    t0 = time.time()
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
        student, comp_stats = apply_module_to_model(model, mod, "auto", rank=32, device=device)
        if student is None:
            print(f"  [FAIL] Injection returned None for {mod}", flush=True)
            results[mod] = "Injection Failed (None)"
            continue

        student = student.half().to(device) if device.startswith("cuda") else student
        student.eval()

        # 1. Forward test
        sample_ids = torch.tensor([[10, 20, 30, 40, 50]], device=device)
        with torch.no_grad():
            out = student(sample_ids)
            logits_shape = tuple(out.logits.shape)
        
        # 2. Evaluation / Perplexity test
        ppl = eval_ppl(student, tokenizer, eval_dataset, eval_args)
        
        # 3. Backward / Gradient test
        student.train()
        dummy_loss = student(sample_ids, labels=sample_ids).loss
        dummy_loss.backward()
        
        dur = time.time() - t0
        print(f"  [OK] {mod} | Forward: {logits_shape} | Eval PPL: {ppl:.2f} | Loss: {dummy_loss.item():.4f} | Time: {dur:.2f}s", flush=True)
        results[mod] = f"PASS (PPL: {ppl:.2f}, Time: {dur:.2f}s)"

        del student, model, out, dummy_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [ERROR] {mod}: {e}", flush=True)
        results[mod] = f"FAIL: {e}"

print("\n" + "=" * 60, flush=True)
print("FINAL SUMMARY OF ALL ADAPTIVE MODULES:", flush=True)
print("=" * 60, flush=True)
for mod, res in results.items():
    print(f"{mod:<30} -> {res}", flush=True)
print("=" * 60, flush=True)

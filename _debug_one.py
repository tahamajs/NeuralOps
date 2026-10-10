import torch, transformers, traceback
from types import SimpleNamespace
from datasets import Dataset
from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
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
    "The quick brown fox jumps over the lazy dog. Scientific reasoning requires proof.",
    "Machine learning models, particularly transformers, rely on self-attention mechanisms.",
]})

base = transformers.AutoModelForCausalLM.from_pretrained("gpt2").half().to(device).eval()
base_ppl = eval_ppl(base, tokenizer, eval_dataset, eval_args)
print(f"Base PPL: {base_ppl:.4f}")

for mod in ["jamba", "hymba_transx", "hla_attention", "granite_hybrid", "halo_regla"]:
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
        student, _ = apply_module_to_model(model, mod, "auto", rank=32, device=device)
        student = student.half().to(device).eval()
        student_ppl = eval_ppl(student, tokenizer, eval_dataset, eval_args)
        delta = student_ppl - base_ppl
        print(f"[{mod}] Student PPL: {student_ppl:.4f}, DELTA: {delta:+.4f} -> {'PASS' if delta > 1.0 else 'FAIL'}")
    except Exception:
        print(f"[{mod}] EXCEPTION:")
        traceback.print_exc()

import torch, transformers
from types import SimpleNamespace
from datasets import Dataset
from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_args = SimpleNamespace(device=device, seq_len=256, eval_stride=128, test_split="test", eval_max_tokens=500)
eval_dataset = Dataset.from_dict({"text": ["The quick brown fox jumps over the lazy dog. Scientific reasoning requires proof."]})

base = transformers.AutoModelForCausalLM.from_pretrained("gpt2").half().to(device).eval()
base_ppl = eval_ppl(base, tokenizer, eval_dataset, eval_args)

samples = ["hybrid_attention", "fused_svd_kernel", "adaptive_sparse_attention", "cosformer", "gqa_attention"]
for m in samples:
    try:
        mod = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
        student, _ = apply_module_to_model(mod, m, "auto", rank=32, device=device)
        student = student.half().to(device).eval()
        ppl = eval_ppl(student, tokenizer, eval_dataset, eval_args)
        delta = ppl - base_ppl
        print(f"[{m}] Base: {base_ppl:.2f}, Student: {ppl:.2f}, Delta: {delta:+.2f} -> {'PASS' if delta > 1.0 else 'FAIL'}")
    except Exception as e:
        print(f"[{m}] Error: {e}")

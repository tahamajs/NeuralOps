import traceback, sys
try:
    import transformers, torch
    from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl
    from types import SimpleNamespace
    from datasets import Dataset

    ds = Dataset.from_dict({'text': ['The quick brown fox jumps over the lazy dog. Scientific testing is required.']})
    args = SimpleNamespace(device='cuda', seq_len=256, eval_stride=128, test_split='test', eval_max_tokens=500)
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    m = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    s, _ = apply_module_to_model(m, 'tri_branch_hybrid', 'auto', rank=32, device='cuda')
    s = s.half().to('cuda').eval()
    print("RUNNING EVAL_PPL tri_branch_hybrid...", flush=True)
    ppl = eval_ppl(s, tok, ds, args)
    print("PPL:", ppl, flush=True)
except Exception as e:
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

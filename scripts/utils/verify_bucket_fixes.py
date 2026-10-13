import transformers, torch
from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl
from types import SimpleNamespace
from datasets import Dataset

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
eval_args = SimpleNamespace(device='cuda', seq_len=256, eval_stride=128, test_split='test', eval_max_tokens=500)
eval_dataset = Dataset.from_dict({'text': [
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning models rely on attention.',
    'Physics describes nature.',
    'Adaptive attention prunes tokens.'
]})

base = transformers.AutoModelForCausalLM.from_pretrained('gpt2').half().to('cuda')
base_ppl = eval_ppl(base, tokenizer, eval_dataset, eval_args)
del base
torch.cuda.empty_cache()
print(f"BASE PPL: {base_ppl:.2f}", flush=True)

samples = ['adaptive_sparse_attention', 'cosformer', 'gqa_attention', 'hybrid_attention']
for mod in samples:
    m = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    s, _ = apply_module_to_model(m, mod, 'auto', rank=32, device='cuda')
    s = s.half().to('cuda')
    s.eval()
    p = eval_ppl(s, tokenizer, eval_dataset, eval_args)
    delta = p - base_ppl
    print(f"MODULE {mod}: PPL={p:.2f} | DELTA={delta:+.2f}", flush=True)
    del s, m
    torch.cuda.empty_cache()

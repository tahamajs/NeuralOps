import torch
import sys
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.eval_all_models_datasets import apply_module_to_model

print('=== Loading GPT-2 ===')
tok = AutoTokenizer.from_pretrained('gpt2')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')

print('=== Injecting gqa_attention ===')
model, stats = apply_module_to_model(model, 'gqa_attention', 'attention', rank=64, device='cuda', tokenizer=tok)
model.cuda().eval()
print('Injection stats:', stats)

print('=== Test generation ===')
prompt = 'The secret passcode is 7402. ' + 'The grass is green. ' * 30 + 'Question: What is the secret passcode? Answer:'
ids = tok.encode(prompt, return_tensors='pt').cuda()

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)

generated = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print('Generated:', repr(generated))
print('Match:', '7402' in generated)

print('=== Baseline (no injection) ===')
model2 = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
with torch.no_grad():
    out2 = model2.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
generated2 = tok.decode(out2[0][ids.shape[1]:], skip_special_tokens=True)
print('Baseline generated:', repr(generated2))
print('Baseline match:', '7402' in generated2)

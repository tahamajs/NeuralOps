import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print("TRANSFORMERS:", transformers.__version__, "| CUDA:", torch.cuda.is_available())
conf = AutoConfig.from_pretrained("/dev/shm/Qwen2.5-0.5B", local_files_only=True)
print("SUCCESS! Config type:", type(conf))

tok = AutoTokenizer.from_pretrained("/dev/shm/Qwen2.5-0.5B", local_files_only=True)
print("SUCCESS! Tokenizer size:", len(tok))

m = AutoModelForCausalLM.from_pretrained("/dev/shm/Qwen2.5-0.5B", torch_dtype=torch.float16, local_files_only=True)
m = m.cuda()
print("SUCCESS! Loaded Qwen model on GPU-3090-1! Total params:", sum(p.numel() for p in m.parameters()))

# Quick forward pass test
x = tok("Antigravity is amazing", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = m(x)
print("Forward pass successful! Logits shape:", out.logits.shape)

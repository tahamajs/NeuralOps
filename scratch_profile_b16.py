import torch
from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model

device = torch.device('cuda')
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('models/gpt2', torch_dtype=torch.float16).to(device)
model.eval()

# batch_size=16, seq_len=1024
B, N = 16, 1024
input_ids = torch.randint(10, 50000, (B, N), dtype=torch.long, device=device)

# 1. Profile full_attention
with torch.no_grad():
    for _ in range(3): model(input_ids)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof_full:
    with torch.no_grad():
        for _ in range(10):
            model(input_ids)

print("\n" + "="*80)
print("=== TOP CUDA KERNELS: FULL_ATTENTION (Softmax) [B=16, N=1024] ===")
print("="*80)
print(prof_full.key_averages().table(sort_by='cuda_time_total', row_limit=15))

# 2. Profile performer_attention
model_perf, _ = apply_module_to_model(model, 'performer_attention', 'auto', 16, device=device)
model_perf.to(dtype=torch.float16, device=device)
model_perf.eval()

with torch.no_grad():
    for _ in range(3): model_perf(input_ids)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof_perf:
    with torch.no_grad():
        for _ in range(10):
            model_perf(input_ids)

print("\n" + "="*80)
print("=== TOP CUDA KERNELS: PERFORMER_ATTENTION [B=16, N=1024] ===")
print("="*80)
print(prof_perf.key_averages().table(sort_by='cuda_time_total', row_limit=15))

# 3. Profile cosformer
model_clean = AutoModelForCausalLM.from_pretrained('models/gpt2', torch_dtype=torch.float16).to(device)
model_cos, _ = apply_module_to_model(model_clean, 'cosformer', 'auto', 16, device=device)
model_cos.to(dtype=torch.float16, device=device)
model_cos.eval()

with torch.no_grad():
    for _ in range(3): model_cos(input_ids)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof_cos:
    with torch.no_grad():
        for _ in range(10):
            model_cos(input_ids)

print("\n" + "="*80)
print("=== TOP CUDA KERNELS: COSFORMER [B=16, N=1024] ===")
print("="*80)
print(prof_cos.key_averages().table(sort_by='cuda_time_total', row_limit=15))

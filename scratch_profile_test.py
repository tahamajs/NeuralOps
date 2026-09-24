import torch
from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model

device = torch.device('cuda')
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('models/gpt2', torch_dtype=torch.float16).to(device)
model.eval()

input_ids = torch.randint(10, 50000, (1, 1024), dtype=torch.long, device=device)

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(input_ids)

print("Profiling full_attention...")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof_full:
    with torch.no_grad():
        for _ in range(10):
            model(input_ids)

print("\n" + "="*50)
print("=== FULL ATTENTION PROFILER ===")
print("="*50)
print(prof_full.key_averages().table(sort_by='cuda_time_total', row_limit=12))

print("\nInjecting performer_attention...")
model_perf, _ = apply_module_to_model(model, 'performer_attention', 'auto', 16, device=device)
model_perf.to(dtype=torch.float16, device=device)
model_perf.eval()

with torch.no_grad():
    for _ in range(5):
        model_perf(input_ids)

print("Profiling performer_attention...")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof_perf:
    with torch.no_grad():
        for _ in range(10):
            model_perf(input_ids)

print("\n" + "="*50)
print("=== PERFORMER ATTENTION PROFILER ===")
print("="*50)
print(prof_perf.key_averages().table(sort_by='cuda_time_total', row_limit=15))

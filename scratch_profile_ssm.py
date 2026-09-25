import os
import sys

# Ensure Codes directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(SCRIPT_DIR, "Codes")
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

import torch
from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model

def profile_module(module_name: str, batch_size: int = 16, seq_len: int = 128, iters: int = 3):
    print(f"\n=======================================================", flush=True)
    print(f"PROFILING SSM MODULE: {module_name} (B={batch_size}, L={seq_len})", flush=True)
    print(f"=======================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("models/gpt2", torch_dtype=torch.float16).to(device)
    model.eval()

    try:
        apply_module_to_model(model, module_name, "gpt2")
        model = model.to(device=device, dtype=torch.float16)
        print(f"Successfully applied {module_name} to GPT-2 on {device} (fp16)", flush=True)
    except Exception as e:
        print(f"Failed to apply {module_name}: {e}", flush=True)
        return

    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(1):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(iters):
                _ = model(dummy_input)
        torch.cuda.synchronize()

    print(f"\n--- TOP 15 CUDA KERNELS BY TIME ({module_name}, {iters} iters, B={batch_size}, L={seq_len}) ---", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)

    total_cuda_calls = sum(item.count for item in prof.key_averages() if item.cuda_time_total > 0)
    print(f"\nTotal CUDA kernel dispatches recorded: {total_cuda_calls}", flush=True)

if __name__ == "__main__":
    module = sys.argv[1] if len(sys.argv) > 1 else "mamba"
    profile_module(module, batch_size=16, seq_len=128, iters=3)

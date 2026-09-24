import os
import sys
import torch
from transformers import AutoModelForCausalLM
from framework.core.model_mod import inject_module

def profile_module(module_name: str, batch_size: int = 16, seq_len: int = 128, iters: int = 3):
    print(f"\n=======================================================", flush=True)
    print(f"PROFILING SSM MODULE: {module_name} (B={batch_size}, L={seq_len})", flush=True)
    print(f"=======================================================", flush=True)

    model = AutoModelForCausalLM.from_pretrained("models/gpt2", torch_dtype=torch.float16).cuda()
    model.eval()

    try:
        inject_module(model, module_name)
    except Exception as e:
        print(f"Failed to inject {module_name}: {e}", flush=True)
        return

    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")

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

    print(f"\n--- TOP 15 CUDA KERNELS BY TIME ({module_name}, {iters} iters) ---", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)

    # Count total CUDA kernel calls
    cuda_events = [e for e in prof.events() if e.is_cuda or e.cuda_time > 0 or e.device_type == torch.profiler.DeviceType.CUDA]
    print(f"\nTotal CUDA events/kernel dispatches recorded: {len(cuda_events)}", flush=True)

if __name__ == "__main__":
    module = sys.argv[1] if len(sys.argv) > 1 else "mamba"
    profile_module(module, batch_size=16, seq_len=128, iters=3)

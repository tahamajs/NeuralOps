import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM

# Ensure Codes directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CODES_DIR = os.path.join(PROJECT_ROOT, "Codes")
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from framework.eval_all_models_datasets import apply_module_to_model

def profile_linformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GPT-2 on {device}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("models/gpt2", torch_dtype=torch.float16).to(device)
    apply_module_to_model(model, "linformer_attention", "gpt2")
    model = model.to(device=device, dtype=torch.float16)
    model.eval()

    seq_lens = [128, 256, 384, 512, 640, 768, 896, 1024]
    batch_size = 16
    iters = 15

    print(f"\nProfiling Linformer across fine-grained sequence lengths (B={batch_size}, iters={iters}):", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for N in seq_lens:
        inp = torch.randint(0, 1000, (batch_size, N), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(inp)
        torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) / iters * 1000.0
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        results[N] = (elapsed_ms, vram_mb)
        print(f"N={N:4d} | Forward: {elapsed_ms:7.2f} ms | VRAM: {vram_mb:7.1f} MB", flush=True)

    print("=" * 60, flush=True)
    # Check jumps
    for i in range(1, len(seq_lens)):
        prev_n, curr_n = seq_lens[i-1], seq_lens[i]
        ratio = results[curr_n][0] / results[prev_n][0]
        len_ratio = curr_n / prev_n
        print(f"Jump {prev_n}->{curr_n} (len {len_ratio:.2f}x): time ratio {ratio:.2f}x", flush=True)

if __name__ == "__main__":
    profile_linformer()

import torch
import gc
from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model

def verify_fix():
    print("=" * 60)
    print("VERIFYING MAMBA_BLOCK FIX ON CUDA")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("1. Loading GPT-2 in FP16...")
    m = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").half().to(device)

    print("2. Injecting mamba_block...")
    m, _ = apply_module_to_model(m, "mamba_block", "attention", rank=64, device=device)
    m.train()

    print("3. Testing forward pass with L=256, B=1...")
    x = torch.randint(0, 50257, (1, 256), device=device)
    out = m(x)

    print(f"4. Output shape: {out.logits.shape}")
    assert out.logits.shape == (1, 256, 50257), f"Shape mismatch: {out.logits.shape}"
    print("   [PASS] Output shape preserved exactly (1, 256, 50257)")

    print("5. Testing backward pass...")
    loss = out.logits.sum()
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_gb = peak_mb / 1024
    print(f"6. Peak VRAM measured: {peak_mb:.2f} MB ({peak_gb:.2f} GB)")

    assert peak_mb < 6000.0, f"Peak VRAM exceeded 6 GB: {peak_mb:.2f} MB"
    print("   [PASS] Peak VRAM is well under 6 GB threshold!")
    print("   [PASS] Training step completed without OOM on Windows!")
    print("=" * 60)

if __name__ == "__main__":
    verify_fix()

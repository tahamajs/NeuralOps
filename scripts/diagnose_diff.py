import sys
import os

# Set ALL offline and threading variables BEFORE importing torch/transformers/huggingface
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from pathlib import Path
_repo = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
_hf_cache = _repo / ".hf_cache"
if _hf_cache.exists():
    os.environ["HF_HOME"] = str(_hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(_hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(_hf_cache)

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

sys.path.insert(0, "Codes")

def diagnose(module_name):
    print(f"Starting diagnosis for {module_name}...", flush=True)
    has_local_model = os.path.exists(r"models\gpt2") or os.path.exists("models/gpt2")
    if has_local_model:
        model_path = r"models\gpt2" if os.path.exists(r"models\gpt2") else "models/gpt2"
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = getattr(cfg, "eos_token_id", 0) or 0
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=cfg, local_files_only=True
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = model.config.eos_token_id or 0
    else:
        from transformers import GPT2Config, GPT2ForSequenceClassification
        cfg = GPT2Config(n_layer=12, n_head=12, n_embd=768, pad_token_id=0)
        model = GPT2ForSequenceClassification(cfg)
    model.eval()

    # Input tensor
    ids = torch.randint(0, 1000, (2, 16))
    attn = torch.ones_like(ids)

    with torch.no_grad():
        baseline_out = model(input_ids=ids, attention_mask=attn).logits.clone()

    # Import module
    mod = None
    try:
        mod = __import__(f"framework.attentions.{module_name}", fromlist=["*"])
    except ModuleNotFoundError:
        mod = __import__(f"framework.compression_engine.{module_name}", fromlist=["*"])

    inject_fn = None
    for name in dir(mod):
        if name.startswith("inject_"):
            inject_fn = getattr(mod, name)
            break

    if inject_fn is None:
        print(f"ERROR: No inject_ function found in {module_name}")
        return

    try:
        inject_fn(model=model, model_config=cfg)
    except TypeError:
        inject_fn(model=model)

    model.eval()

    with torch.no_grad():
        injected_out = model(input_ids=ids, attention_mask=attn).logits

    max_diff = (baseline_out - injected_out).abs().max().item()
    mean_diff = (baseline_out - injected_out).abs().mean().item()
    print(f"=== DIAGNOSTIC REPORT: {module_name} ===")
    print(f"init_diff (max):  {max_diff:.6f}")
    print(f"init_diff (mean): {mean_diff:.6f}")

    # Inspect first attention layer
    first_attn = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        first_layer = model.transformer.h[0]
        for attr in ("attn", "attention", "self_attn", "self"):
            if hasattr(first_layer, attr):
                first_attn = getattr(first_layer, attr)
                break
    
    if first_attn is not None:
        print(f"First attention layer class: {first_attn.__class__.__name__}")
        has_lora_b = False
        for name, param in first_attn.named_parameters():
            if "lora_b" in name.lower() or "lora_B" in name:
                has_lora_b = True
                val = param.abs().max().item()
                print(f"  {name} max_abs = {val:.8f} (zero={val < 1e-6})")
            elif "weight" in name and "lora" not in name.lower():
                print(f"  {name}.norm() = {param.norm().item():.4f}, shape = {tuple(param.shape)}")
        if not has_lora_b:
            print("  No lora_B parameter found in attention layer.")
    else:
        print("Could not locate first attention layer.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_diff.py <module_name>")
        sys.exit(1)
    try:
        diagnose(sys.argv[1])
    except Exception as e:
        import traceback
        traceback.print_exc()

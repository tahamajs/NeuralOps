import sys, os, json, torch
from pathlib import Path

repo_root = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
hf_cache = repo_root / ".hf_cache"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if hf_cache.exists():
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
sys.path.insert(0, str(repo_root / "Codes"))

from transformers import AutoConfig, AutoModelForSequenceClassification
import framework.attentions.gqa_attention as gqa_mod

models = {
    "distilbert": str(repo_root / "models" / "distilbert_sst2"),
    "roberta": str(repo_root / "models" / "roberta-base"),
    "gpt2": str(repo_root / "models" / "gpt2")
}

results = {}
for name, path in models.items():
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = getattr(cfg, "eos_token_id", 0) or 0
    model = AutoModelForSequenceClassification.from_pretrained(path, config=cfg, local_files_only=True)
    model.eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 1000, (2, 16))
    attn = torch.ones_like(ids)

    with torch.no_grad():
        before = model(input_ids=ids, attention_mask=attn).logits.clone()

    gqa_mod.inject_gqa(model=model, model_config=cfg)

    with torch.no_grad():
        after = model(input_ids=ids, attention_mask=attn).logits.clone()

    diff = (before - after).abs().max().item()
    results[name] = diff
    print(f"MODEL {name:12s} diff={diff:.6f}")

with open(repo_root / "gqa_test_result.json", "w") as f:
    json.dump(results, f, indent=2)

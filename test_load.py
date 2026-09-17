import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModel, AutoConfig

print("Testing offline load...")
for name, p in [("DistilBERT", r"models\distilbert_sst2"), ("RoBERTa", r"models\roberta-base"), ("GPT2", r"models\gpt2")]:
    try:
        cfg = AutoConfig.from_pretrained(p, local_files_only=True)
        m = AutoModel.from_pretrained(p, config=cfg, local_files_only=True)
        print(f"{name} loaded successfully from {p}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

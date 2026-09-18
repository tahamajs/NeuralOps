import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

model_path = "models/distilbert_sst2"
ckpt_path = "models/base_checkpoints/distilbert_agnews/finetuned_model.pt"
out_dir = Path("models/base_checkpoints/distilbert_agnews")

print("Loading base model...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
tok = AutoTokenizer.from_pretrained(model_path)

print(f"Loading trained weights from {ckpt_path}...", flush=True)
st = torch.load(ckpt_path, map_location="cpu")
# If it's a head checkpoint, load head into model
model.load_state_dict(st, strict=False)

print("Saving complete state_dict...", flush=True)
torch.save(model.state_dict(), ckpt_path)

print("Saving HuggingFace pretrained format...", flush=True)
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("SUCCESS: distilbert_agnews fully saved.", flush=True)

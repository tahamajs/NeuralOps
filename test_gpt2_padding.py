import sys, os
os.chdir(r"D:\SVD_Project")
sys.path.insert(0, r"D:\SVD_Project")
sys.path.insert(0, r"D:\SVD_Project\Codes")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

model_path = r"models\gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

data_dir = r"Codes\framework\data\glue_data\sst2"
ds = load_from_disk(data_dir)
val = ds["validation"]
texts = val["sentence"][:50]
labels = torch.tensor(val["label"][:50], device="cuda")

model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.config.pad_token_id = tokenizer.pad_token_id
ckpt = torch.load(r"results\gpt2_sst2_lora_baseline\finetuned_model.pt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.cuda().eval()

# 1. Left padding
tokenizer.padding_side = "left"
inputs_left = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out_left = model(**inputs_left).logits
    acc_left = (out_left.argmax(dim=-1) == labels).float().mean().item() * 100.0
print(f"GPT2 baseline accuracy with LEFT padding: {acc_left:.2f}%")

# 2. Right padding
tokenizer.padding_side = "right"
inputs_right = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out_right = model(**inputs_right).logits
    acc_right = (out_right.argmax(dim=-1) == labels).float().mean().item() * 100.0
print(f"GPT2 baseline accuracy with RIGHT padding: {acc_right:.2f}%")

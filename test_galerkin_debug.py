import sys, os
os.chdir(r"D:\SVD_Project")
sys.path.insert(0, r"D:\SVD_Project")
sys.path.insert(0, r"D:\SVD_Project\Codes")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

model_path = r"models\distilbert_sst2"
tokenizer = AutoTokenizer.from_pretrained(model_path)

data_dir = r"Codes\framework\data\glue_data\sst2"
ds = load_from_disk(data_dir)
val = ds["validation"]
texts = val["sentence"][:50]
labels = torch.tensor(val["label"][:50], device="cuda")

model = AutoModelForSequenceClassification.from_pretrained(model_path)
ckpt = torch.load(r"results\unified_asvd\distilbert\finetuned_model.pt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.cuda().eval()

from framework.core.attention_utils import inject_attention
inject_attention(model, "galerkin", copy_weights=True)

# 1. Left padding
tokenizer.padding_side = "left"
inputs_left = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    inj_logits_left = model(**inputs_left).logits
    acc_left = (inj_logits_left.argmax(dim=-1) == labels).float().mean().item() * 100.0
print(f"Galerkin accuracy with LEFT padding: {acc_left:.2f}%")

# 2. Right padding
tokenizer.padding_side = "right"
inputs_right = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    inj_logits_right = model(**inputs_right).logits
    acc_right = (inj_logits_right.argmax(dim=-1) == labels).float().mean().item() * 100.0
print(f"Galerkin accuracy with RIGHT padding: {acc_right:.2f}%")

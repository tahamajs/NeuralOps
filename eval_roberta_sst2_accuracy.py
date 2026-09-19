import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

model_path = r"D:\svd_project\models\roberta_sst2"
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cuda")

val = load_from_disk(r"D:\svd_project\Codes\framework\data\glue_data\sst2")["validation"]
print(f"Evaluating first 100 samples on SST-2 validation...")

corr = 0
preds = []
for i in range(100):
    inp = tokenizer(val[i]["sentence"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    p = out.logits.argmax(-1).item()
    preds.append(p)
    if p == val[i]["label"]:
        corr += 1

print(f"Accuracy on 100 samples: {corr}/100 = {corr}%")
print(f"Predicted class distribution: 0s={preds.count(0)}, 1s={preds.count(1)}")

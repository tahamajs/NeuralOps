import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

model_path = 'D:/SVD_Project/models/distilbert_sst2'
print(f'Loading checkpoint from: {model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
model.eval()

ds = load_dataset('glue', 'sst2', split='validation[:500]')
sentences = list(ds['sentence'])
labels = list(ds['label'])
enc = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to('cuda')
with torch.no_grad():
    logits = model(**enc).logits
preds = torch.argmax(logits, dim=-1).cpu().tolist()
acc = accuracy_score(labels, preds) * 100
print(f'GPU distilbert_sst2 Validation Accuracy (N=500): {acc:.2f}%')

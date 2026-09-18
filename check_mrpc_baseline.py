import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

model_id = 'textattack/roberta-base-MRPC'
print('Loading', model_id)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
model.eval()

ds = load_dataset('glue', 'mrpc', split='validation[:200]')
s1 = list(ds['sentence1'])
s2 = list(ds['sentence2'])
labels = list(ds['label'])
enc = tok(s1, s2, padding=True, truncation=True, return_tensors='pt').to('cuda')
with torch.no_grad():
    logits = model(**enc).logits
preds = torch.argmax(logits, dim=-1).cpu().tolist()
acc = accuracy_score(labels, preds) * 100
print(f'{model_id} Accuracy on MRPC: {acc:.2f}%')

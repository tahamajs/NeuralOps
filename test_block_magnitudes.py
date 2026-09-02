import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from framework.eval_all_models_datasets import apply_module_to_model

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSequenceClassification.from_pretrained('t5-small')
apply_module_to_model(model, 'jamba', 'attention')
model.to(device).half()

ds = load_from_disk('D:/SVD_Project/Codes/framework/data/glue_data/sst2')['train']
texts = [ds[0]['sentence'], ds[1]['sentence']]
enc = tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
ids = enc['input_ids'].to(device)
mask = enc['attention_mask'].to(device)

for idx, b in enumerate(model.transformer.encoder.block):
    def hook(m, inp, out, idx=idx):
        h = inp[0]
        print(f"Encoder Block {idx} input max abs: {h.abs().max().item():.2f}")
    b.register_forward_hook(hook)

out = model(input_ids=ids, attention_mask=mask)
print("Forward finished!")

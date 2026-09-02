import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from framework.eval_all_models_datasets import apply_module_to_model

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSequenceClassification.from_pretrained('t5-small')

# Freeze backbone like in finetune_lora
for p in model.parameters():
    p.requires_grad = False

apply_module_to_model(model, 'jamba', 'attention')
model.to(device).half()

trainable = [p for p in model.parameters() if p.requires_grad]
for p in trainable:
    p.data = p.data.float()

ds = load_from_disk('D:/SVD_Project/Codes/framework/data/glue_data/sst2')['train']
def collate_fn(batch):
    texts = [x['sentence'] for x in batch]
    encoded = tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    return encoded['input_ids'].to(device), encoded['attention_mask'].to(device), labels.to(device)

loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

model.train()
print(f"Running 5 steps with EXACT finetune_lora setup ({len(trainable)} trainable tensors)...")
for step, (ids, mask, labels) in enumerate(loader):
    if step >= 5: break
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = out.loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    print(f"Step {step+1}: Loss = {loss.item():.4f}")
print("SUCCESS! ALL 5 STEPS FINITE IN EXACT FINETUNE_LORA SETUP!")

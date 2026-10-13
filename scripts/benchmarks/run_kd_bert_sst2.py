
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from neuralops.core._attention_injectors import apply_module_to_model

def run_kd_sst2(module_name='block_transformer_hiba', rank=32, steps=60):
    device = torch.device('cpu')
    teacher_name = 'textattack/bert-base-uncased-SST-2'
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name).to(device)
    teacher.eval()
    
    student = AutoModelForSequenceClassification.from_pretrained(teacher_name).to(device)
    print(f'Injecting {module_name} into BERT...')
    student, stats = apply_module_to_model(student, module_name, 'attention', rank=rank, device=device)
    
    ds = load_dataset('glue', 'sst2')
    train_ds = ds['train'].select(range(min(len(ds['train']), 500)))
    val_ds = ds['validation']
    
    def tokenize(b):
        return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=128)
    
    train_enc = train_ds.map(tokenize, batched=True)
    val_enc = val_ds.map(tokenize, batched=True)
    train_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    val_loader = DataLoader(val_enc, batch_size=32)
    def evaluate(m):
        m.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for b in val_loader:
                out = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                corr += (out.logits.argmax(-1) == b['label'].to(device)).sum().item()
                tot += len(b['label'])
        return corr / tot

    pre_acc = evaluate(student)
    print(f'Pre-KD Student SST-2 Accuracy: {pre_acc * 100:.2f}%')
    
    train_loader = DataLoader(train_enc, batch_size=8, shuffle=True)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    ce_loss_fn = nn.CrossEntropyLoss()
    
    student.train()
    step = 0
    print('Starting Knowledge Distillation on SST-2...')
    for epoch in range(10):
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)
            
            with torch.no_grad():
                t_out = teacher(input_ids=ids, attention_mask=mask)
            s_out = student(input_ids=ids, attention_mask=mask)
            
            loss_ce = ce_loss_fn(s_out.logits, lbl)
            loss_kl = kl_loss_fn(
                torch.log_softmax(s_out.logits / 2.0, dim=-1),
                torch.softmax(t_out.logits / 2.0, dim=-1)
            ) * 4.0
            loss = 0.5 * loss_ce + 0.5 * loss_kl
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f'Step {step}/{steps} - Loss: {loss.item():.4f}')
            if step >= steps:
                break
        if step >= steps:
            break
            
    post_acc = evaluate(student)
    print(f'Post-KD Student SST-2 Accuracy: {post_acc * 100:.2f}%')

if __name__ == '__main__':
    run_kd_sst2()

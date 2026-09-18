import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import apply_module_to_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True, help="Module to inject")
    parser.add_argument("--base_model", type=str, default="D:/SVD_Project/models/gpt2_agnews_base", help="Path to base finetuned model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="D:/SVD_Project/results/kl_distill")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer and base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.base_model).to(device)
    teacher_model.eval()
    
    student_model = AutoModelForSequenceClassification.from_pretrained(args.base_model).to(device)
    student_model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Injecting {args.module} into student model...")
    apply_module_to_model(student_model, args.module, "gpt")
    
    # Freeze base model, only train LoRA / injected params
    for name, param in student_model.named_parameters():
        if "lora" not in name and args.module not in name.lower() and "adapter" not in name.lower():
            param.requires_grad = False
    
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_func, batched=True)
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr)
    
    out_dir = os.path.join(args.output_dir, args.module)
    os.makedirs(out_dir, exist_ok=True)
    
    print("Starting distillation...")
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)
    
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits
                
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits
            
            # KL divergence: D_KL(P || Q) = sum P * log(P/Q) = sum P * (log P - log Q)
            # PyTorch KLDivLoss expects input as log-probabilities (student) and target as probabilities (teacher)
            distillation_loss = kl_loss_fn(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1)
            )
            
            # Optionally add hard label loss (alpha * CE + (1-alpha) * KL)
            ce_loss = F.cross_entropy(student_logits, batch["label"])
            loss = 0.5 * distillation_loss + 0.5 * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} average loss: {total_loss / len(train_dataloader)}")
        
    print(f"Saving distilled model to {out_dir}")
    student_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()

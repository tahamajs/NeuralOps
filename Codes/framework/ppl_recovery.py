import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import pandas as pd

sys.path.append("D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import apply_module_to_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, default="D:/SVD_Project/models/gpt2")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--out_csv", type=str, default="D:/SVD_Project/results/recovery_ppl.csv")
    return parser.parse_args()

def calc_ppl(model, tokenizer, encodings, stride=512, max_length=1024, max_batches=None):
    device = next(model.parameters()).device
    model.eval()
    nlls = []
    prev_end_loc = 0
    seq_len = encodings.input_ids.size(1)
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
        if max_batches and len(nlls) >= max_batches:
            break
            
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    # For training, let's tokenize and chunk
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    train_ds = ds.select(range(min(1000, len(ds)))).map(tokenize_func, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=4)
    
    test_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(test_ds["text"])
    test_encodings = tokenizer(test_text, return_tensors="pt")
    
    results = []
    
    for mod in args.modules:
        print(f"\n=== Testing {mod} ===")
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        apply_module_to_model(model, mod, "gpt")
        
        # Zero-shot PPL
        try:
            zero_shot = calc_ppl(model, tokenizer, test_encodings)
        except Exception as e:
            print(f"Error zero-shot PPL for {mod}: {e}")
            zero_shot = float('nan')
            
        print(f"Zero-shot PPL: {zero_shot}")
        
        # Freeze all except adapter/lora/mod
        trainable = 0
        for name, param in model.named_parameters():
            if "lora" not in name and mod not in name.lower() and "adapter" not in name.lower() and "proj" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable += param.numel()
                
        print(f"Trainable params: {trainable}")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        
        model.train()
        step = 0
        pbar = tqdm(total=args.steps, desc=f"Recovering {mod}")
        while step < args.steps:
            for batch in train_dataloader:
                if step >= args.steps:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = input_ids.clone()
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
                step += 1
        pbar.close()
        
        # Post-recovery PPL
        try:
            post_rec = calc_ppl(model, tokenizer, test_encodings)
        except Exception as e:
            print(f"Error post-recovery PPL for {mod}: {e}")
            post_rec = float('nan')
            
        print(f"Post-recovery PPL: {post_rec}")
        
        results.append({
            "module": mod,
            "zero_shot_ppl": zero_shot,
            "post_recovery_ppl": post_rec,
            "delta": zero_shot - post_rec if not pd.isna(post_rec) and not pd.isna(zero_shot) else float('nan')
        })
        
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()

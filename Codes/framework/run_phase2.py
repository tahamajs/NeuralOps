import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from framework.core.attention_utils import inject_attention
from framework.models_conf.core_loaders import load_model

def run_for_model(model_path, dataset_name, seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 2 for {model_path} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    ds = load_from_disk(f"D:/SVD_Project/Codes/framework/data/glue_data/{dataset_name}")
    train_ds = ds["train"]
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        
    from framework.eval_all_models_datasets import apply_module_to_model
    model, _ = apply_module_to_model(model, "a3")
    
    from framework.utils.lora import apply_lora
    model, _ = apply_lora(model, r=8, alpha=32)
    model = model.to(device)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4)
    model.train()
    
    from framework.run_quality_sweep import encode
    curr_batch_size = 16
    
    losses = {}
    
    step = 0
    while step <= 100:
        i = (step * curr_batch_size) % max(1, len(train_ds) - curr_batch_size)
        batch = {k: train_ds[k][i:i+curr_batch_size] for k in train_ds.column_names}
        bx, by = encode(tokenizer, batch, dataset_name, seq_len, device)
        
        opt.zero_grad()
        out = model(**bx, labels=by)
        loss = out.loss
        loss.backward()
        opt.step()
        
        if step in [0, 10, 25, 50, 100]:
            print(f"Step {step} Loss: {loss.item():.4f}")
            losses[step] = loss.item()
            
        step += 1
        
    print(f"Final losses for {model_path}: {losses}")
    return losses

def main():
    run_for_model("D:/svd_project/models/distilbert_sst2", "sst2", 128)
    run_for_model("D:/svd_project/models/roberta-base", "sst2", 128)
    run_for_model("D:/svd_project/models/gpt2", "sst2", 128)

if __name__ == "__main__":
    main()

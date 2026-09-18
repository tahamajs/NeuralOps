import os
import sys
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

def encode(tokenizer, batch, dataset_name, seq_len, device):
    def fields(dataset):
        if dataset in ["sst2", "cola", "imdb", "ag_news"]: return "text", None
        elif dataset in ["mnli", "mnli-mm"]: return "premise", "hypothesis"
        elif dataset in ["qqp", "mrpc", "rte"]: return "sentence1", "sentence2"
        elif dataset == "qnli": return "question", "sentence"
        else: return "text", None
        
    a, b = fields(dataset_name)
    kwargs = {"text": batch[a]}
    if b is not None: kwargs["text_pair"] = batch[b]
    
    enc = tokenizer(**kwargs, max_length=seq_len, padding="max_length", truncation=True, return_tensors="pt")
    
    label_key = "labels" if "labels" in batch else "label"
    labels = torch.tensor(batch[label_key], dtype=torch.long)
    
    res = {k: v.to(device) for k, v in enc.items()}
    res["labels"] = labels.to(device)
    return res

def evaluate(model, tokenizer, ds, dataset_name, seq_len, batch_size, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = ds[i:i+batch_size]
            enc = encode(tokenizer, batch, dataset_name, seq_len, device)
            
            outputs = model(**enc)
            logits = outputs.logits
            
            preds = logits.argmax(dim=-1)
            correct += (preds == enc["labels"]).sum().item()
            total += len(enc["labels"])
            
    return (correct / total) * 100

def main():
    if len(sys.argv) < 4:
        print("Usage: eval_int8_quantized.py <model_path> <dataset_name> <seq_len>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    dataset_name = sys.argv[2]
    seq_len = int(sys.argv[3])
    
    device = "cpu" # Int8 CPU quantization
    
    # Load dataset
    dataset_dirs = [
        f"D:/SVD_Project/Codes/framework/data/glue_data/{dataset_name}",
        f"D:/svd_project/Codes/framework/data/glue_data/{dataset_name}"
    ]
    
    dataset_dir = None
    for d in dataset_dirs:
        if os.path.exists(d):
            dataset_dir = d
            break
            
    if dataset_dir is None:
        print(f"Dataset {dataset_name} not found!")
        sys.exit(1)
        
    print(f"Loading dataset from {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    val_ds = ds["validation"]
    
    # Check if MNLI has matched/mismatched
    if dataset_name == "mnli":
        val_ds = ds["validation_matched"]
        
    print(f"Loading {model_path} on {device}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # baseline metrics
    start = time.time()
    baseline_acc = evaluate(model, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    baseline_time = time.time() - start
    
    print(f"Baseline Acc: {baseline_acc:.2f}%, Time: {baseline_time:.2f}s")
    
    print("Applying dynamic int8 quantization...")
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    start = time.time()
    q_acc = evaluate(qmodel, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    q_time = time.time() - start
    
    # get sizes
    def get_size(m):
        torch.save(m.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        os.remove("temp.p")
        return size
        
    baseline_size = get_size(model) / 1024**2
    q_size = get_size(qmodel) / 1024**2
    
    print(f"Quantized Acc: {q_acc:.2f}%, Time: {q_time:.2f}s")
    print(f"Baseline Size: {baseline_size:.2f} MB")
    print(f"Quantized Size: {q_size:.2f} MB")

if __name__ == '__main__':
    main()

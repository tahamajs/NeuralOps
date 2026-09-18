import torch
import time
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from framework.core.attention_utils import inject_attention

def _safe_load_state(target_model, raw_state):
    st = raw_state.get("module_state_dict", raw_state.get("state_dict", raw_state)) if isinstance(raw_state, dict) else raw_state
    
    # Check if there's a lora state dict
    lora_st = raw_state.get("lora_state_dict", {}) if isinstance(raw_state, dict) else {}
    if lora_st:
        from framework.utils.lora import apply_lora
        apply_lora(target_model, r=8, alpha=32)
        target_model.load_state_dict(lora_st, strict=False)

    model_st = target_model.state_dict()
    filtered_st = {}
    for k, v in st.items():
        if k in model_st:
            if v.shape != model_st[k].shape:
                continue
            filtered_st[k] = v
    target_model.load_state_dict(filtered_st, strict=False)

def encode(tokenizer, batch, dataset_name, seq_len, device):
    def fields(dataset):
        if dataset in ["sst2", "cola", "imdb"]: return "sentence", None
        elif dataset in ["ag_news"]: return "text", None
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

def get_size(m):
    torch.save(m.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    os.remove("temp.p")
    return size / 1024**2

def main():
    if len(sys.argv) < 5:
        print("Usage: eval_int8.py <base_model_path> <chkpt_path> <dataset_name> <seq_len>")
        sys.exit(1)
        
    base_model_path = sys.argv[1]
    chkpt_path = sys.argv[2]
    dataset_name = sys.argv[3]
    seq_len = int(sys.argv[4])
    
    device = "cpu"
    
    dataset_dir = f"D:/svd_project/Codes/framework/data/glue_data/{dataset_name}"
    ds = load_from_disk(dataset_dir)
    val_ds = ds["validation"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Baseline
    print("Loading baseline...")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        
    base_size = get_size(base_model)
    print("Evaluating Baseline...")
    base_acc = evaluate(base_model, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    
    print("Quantizing Baseline INT8...")
    q_base = torch.ao.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
    q_base_size = get_size(q_base)
    q_base_acc = evaluate(q_base, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    
    # A3 Model
    print("Loading A3...")
    a3_model = AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)
    if tokenizer.pad_token_id is not None:
        a3_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Inject A3
    from framework.eval_all_models_datasets import apply_module_to_model
    a3_model, _ = apply_module_to_model(a3_model, "a3")
    
    if os.path.exists(chkpt_path):
        print(f"Loading weights from {chkpt_path}")
        raw_state = torch.load(chkpt_path, map_location="cpu")
        _safe_load_state(a3_model, raw_state)
    
    a3_size = get_size(a3_model)
    print("Evaluating A3...")
    a3_acc = evaluate(a3_model, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    
    print("Quantizing A3 INT8...")
    # Also quantize A3FactorizedLinear if needed, but standard dynamic quantization only does nn.Linear
    # Which is exactly what we want: compress the fallback layers and keep the already compressed factorized layers intact
    q_a3 = torch.ao.quantization.quantize_dynamic(a3_model, {torch.nn.Linear}, dtype=torch.qint8)
    q_a3_size = get_size(q_a3)
    q_a3_acc = evaluate(q_a3, tokenizer, val_ds, dataset_name, seq_len, 32, device)
    
    print("="*40)
    print(f"Baseline FP32: {base_size:.2f} MB, Acc: {base_acc:.2f}%")
    print(f"Baseline INT8: {q_base_size:.2f} MB, Acc: {q_base_acc:.2f}%")
    print(f"A3 FP32: {a3_size:.2f} MB, Acc: {a3_acc:.2f}%")
    print(f"A3 INT8: {q_a3_size:.2f} MB, Acc: {q_a3_acc:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()

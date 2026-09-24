import math
import torch
import time

@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, ds, seq_len, batch_size, device, fp16=True):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    t0 = time.monotonic()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    # Text column for wikitext is 'text'
    texts = [t for t in ds["text"] if len(t.strip()) > 0]
    
    # Tokenize everything
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    # Create sliding windows or simple chunks
    stride = seq_len
    
    for i in range(0, input_ids.size(0), stride):
        begin_loc = i
        end_loc = min(i + seq_len, input_ids.size(0))
        trg_len = end_loc - i
        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        target_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=fp16):
            outputs = model(input_chunk, labels=target_chunk)
            loss = outputs.loss
            
        total_nll += loss.item() * trg_len
        total_tokens += trg_len

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    t1 = time.monotonic()
    
    return {
        "status": "PASS" if ppl < 100.0 else "FAIL",
        "accuracy_pct": 0.0,
        "f1_score": 0.0,
        "perplexity": ppl,
        "throughput": total_tokens / (t1 - t0) if t1 > t0 else 0,
        "vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device_type == "cuda" else 0
    }
def evaluate_comprehensive(
    model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None
):
    model.eval()
    correct = total = 0
    correct_gpu = torch.tensor(0, device=device, dtype=torch.long)
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    batches = (

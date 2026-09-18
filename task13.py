import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "D:/SVD_Project/models/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def calc_ppl(ds_name, split, seq_len_limit, stride):
    ds = load_dataset("Salesforce/wikitext", ds_name, split=split)
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + seq_len_limit, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()

print("a. 1024/1024:", calc_ppl("wikitext-2-raw-v1", "test", 1024, 1024))
print("b. 512/512:", calc_ppl("wikitext-2-raw-v1", "test", 512, 512))
print("c. 2048/2048:", calc_ppl("wikitext-2-raw-v1", "test", 2048, 2048))
print("d. wikitext-103-raw-v1 (1024/512):", calc_ppl("wikitext-103-raw-v1", "test", 1024, 512))

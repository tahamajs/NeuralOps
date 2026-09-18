import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append("D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import apply_module_to_model

print("=== TASK 1.1 ===")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print("Number of rows:", len(ds))
text = "\n\n".join(ds["text"])
print("Total character count:", len(text))
print("First 200 characters:", repr(text[:200]))

print("\n=== TASK 1.2 ===")
tokenizer = AutoTokenizer.from_pretrained("D:/SVD_Project/models/gpt2")
tokens = tokenizer("Hello world")
print(tokens)

print("\n=== TASK 1.4 ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "D:/SVD_Project/models/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("Official HF PPL:", ppl.item())

print("\n=== TASK 2.1 ===")
model2 = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2')
before_keys = set(model2.state_dict().keys())
before = sum(p.sum().item() for p in model2.parameters())
apply_module_to_model(model2, 'moba', "gpt")
after_keys = set(model2.state_dict().keys())
after = sum(p.sum().item() for p in model2.parameters())
print(f'Before: {before}, After: {after}, Diff: {after-before}')
print(f'Keys added: {after_keys - before_keys}')

print("\n=== TASK 4.1 ===")
model4 = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2').eval()
inputs = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    baseline = model4(inputs).logits
apply_module_to_model(model4, 'gqa', "gpt")
with torch.no_grad():
    injected = model4(inputs).logits
print(f'GQA Max diff: {(baseline - injected).abs().max().item()}')


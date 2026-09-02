import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from framework.eval_all_models_datasets import apply_module_to_model

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSequenceClassification.from_pretrained('t5-small')
apply_module_to_model(model, 'jamba', 'attention')
model.to(device).half()

# Register forward hooks on EVERY submodule to catch the EXACT first layer that produces NaN
def make_hook(name):
    def hook(module, input, output):
        # Check input
        for idx, inp in enumerate(input):
            if isinstance(inp, torch.Tensor) and not torch.isfinite(inp).all():
                print(f"[FIRST INPUT NAN] {name} | Arg {idx} has NaN/Inf!")
                sys.exit(0)
        # Check output
        if isinstance(output, tuple):
            out_tensors = [t for t in output if isinstance(t, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            out_tensors = [output]
        else:
            out_tensors = []
        for idx, out in enumerate(out_tensors):
            if not torch.isfinite(out).all():
                print(f"[FIRST OUTPUT NAN] {name} | Output {idx} has NaN/Inf! Shape: {out.shape}")
                # Print module attributes or details
                sys.exit(0)
    return hook

for name, mod in model.named_modules():
    mod.register_forward_hook(make_hook(name))

ds = load_from_disk('D:/SVD_Project/Codes/framework/data/glue_data/sst2')['train']
texts = [ds[0]['sentence'], ds[1]['sentence']]
encoded = tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
ids = encoded['input_ids'].to(device)
mask = encoded['attention_mask'].to(device)
labels = torch.tensor([ds[0]['label'], ds[1]['label']], device=device)

print("Starting forward pass with forward hook probe...")
out = model(input_ids=ids, attention_mask=mask, labels=labels)
print("Forward pass completed without NaN? Loss =", out.loss)

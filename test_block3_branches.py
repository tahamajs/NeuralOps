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

ds = load_from_disk('D:/SVD_Project/Codes/framework/data/glue_data/sst2')['train']
enc = tokenizer([ds[0]['sentence'], ds[1]['sentence']], padding='max_length', max_length=64, truncation=True, return_tensors='pt')
ids = enc['input_ids'].to(device)
mask = enc['attention_mask'].to(device)

block3 = model.transformer.encoder.block[3].layer[0].SelfAttention
def mamba_hook(m, inp, out):
    print("Block 3 Mamba out max abs:", out[0].abs().max().item())
def norm1_hook(m, inp, out):
    print("Block 3 Norm1 out max abs:", out.abs().max().item())
def attn_hook(m, inp, out):
    print("Block 3 Attn proj out max abs:", out.abs().max().item())
def norm2_hook(m, inp, out):
    print("Block 3 Norm2 out max abs:", out.abs().max().item())
def moe_hook(m, inp, out):
    print("Block 3 MoE out max abs:", out.abs().max().item())
def norm3_hook(m, inp, out):
    print("Block 3 Norm3 out max abs:", out.abs().max().item())

block3.mamba.register_forward_hook(mamba_hook)
block3.norm1.register_forward_hook(norm1_hook)
block3.out_lin.register_forward_hook(attn_hook)
block3.norm2.register_forward_hook(norm2_hook)
block3.moe.register_forward_hook(moe_hook)
block3.norm3.register_forward_hook(norm3_hook)

out = model(input_ids=ids, attention_mask=mask)

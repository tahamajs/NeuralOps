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

# Intercept inside block 3 Jamba2Block
jamba_b3 = model.transformer.encoder.block[3].layer[0].SelfAttention

orig_forward = jamba_b3.forward
def debug_forward(*args, **kwargs):
    hidden_states = args[0] if args else kwargs['hidden_states']
    print("b3 hidden_states max:", hidden_states.abs().max().item(), "has_nan:", torch.isnan(hidden_states).any().item())
    
    # Branch 1: Mamba
    mamba_out, _ = jamba_b3.mamba(hidden_states, attention_mask=kwargs.get('attention_mask'))
    print("b3 mamba_out max:", mamba_out.abs().max().item(), "has_nan:", torch.isnan(mamba_out).any().item())
    
    x1 = jamba_b3.norm1(hidden_states + mamba_out)
    print("b3 x1 max:", x1.abs().max().item(), "has_nan:", torch.isnan(x1).any().item())
    
    # Branch 2: MHA
    B, L, D = hidden_states.shape
    q = jamba_b3.q_lin(x1).view(B, L, jamba_b3.n_heads, jamba_b3.head_dim).transpose(1, 2)
    k = jamba_b3.k_lin(x1).view(B, L, jamba_b3.n_heads, jamba_b3.head_dim).transpose(1, 2)
    v = jamba_b3.v_lin(x1).view(B, L, jamba_b3.n_heads, jamba_b3.head_dim).transpose(1, 2)
    scale = 1.0 / (jamba_b3.head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    print("b3 scores max:", scores.abs().max().item(), "has_nan:", torch.isnan(scores).any().item())
    
    attn_mask = kwargs.get('attention_mask')
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            scores = scores + attn_mask.clamp(min=-10000.0).to(scores.dtype)
    probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    probs = torch.nan_to_num(probs, nan=0.0)
    print("b3 probs max:", probs.abs().max().item(), "has_nan:", torch.isnan(probs).any().item())
    
    attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, jamba_b3.all_head_size)
    attn_proj = jamba_b3.out_lin(attn_out)
    print("b3 attn_proj max:", attn_proj.abs().max().item(), "has_nan:", torch.isnan(attn_proj).any().item())
    
    x2 = jamba_b3.norm2(x1 + attn_proj)
    print("b3 x2 max:", x2.abs().max().item(), "has_nan:", torch.isnan(x2).any().item())
    
    # Branch 3: MoE
    moe_out = jamba_b3.moe(x2)
    print("b3 moe_out max:", moe_out.abs().max().item(), "has_nan:", torch.isnan(moe_out).any().item())
    
    x3 = jamba_b3.norm3(x2 + moe_out)
    print("b3 x3 max:", x3.abs().max().item(), "has_nan:", torch.isnan(x3).any().item())
    return x3, None

jamba_b3.forward = debug_forward
out = model(input_ids=ids, attention_mask=mask)

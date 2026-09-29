#!/usr/bin/env python3
import copy
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Codes"))

from framework.eval_all_models_datasets import apply_module_to_model

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tok = AutoTokenizer.from_pretrained("models/gpt2_wikitext")
model = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").cuda().half()
student, _ = apply_module_to_model(copy.deepcopy(model), "triton_svd_linear_attention", "attention", rank=64, device="cuda")
student.eval()

text = "\n\n".join(ds["text"])
enc = tok(text, return_tensors="pt")
ids = enc.input_ids[:, 768:1024].cuda()

print("Tracing window [768:1024] through transformer blocks...")
with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        hidden_states = student.transformer.wte(ids) + student.transformer.wpe(torch.arange(256, device="cuda"))
        hidden_states = student.transformer.drop(hidden_states)
        
        for i, block in enumerate(student.transformer.h):
            norm1 = block.ln_1(hidden_states)
            attn_mod = block.attn._wrapped_attention
            
            # Trace inner attn
            B, N, C = norm1.shape
            q = attn_mod.q_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
            k = attn_mod.k_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
            v = attn_mod.v_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
            
            from framework.core.triton_kernels import triton_dispatch
            attn_out = triton_dispatch(attn_mod.triton_kernel, q, k, v, is_causal=True, **attn_mod.triton_kwargs)
            
            has_nan_attn = torch.isnan(attn_out).any().item()
            has_inf_attn = torch.isinf(attn_out).any().item()
            
            out_proj = attn_mod.out_proj(attn_out.transpose(1, 2).contiguous().view(B, N, C))
            hidden_states = hidden_states + out_proj
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            
            has_nan_block = torch.isnan(hidden_states).any().item()
            print(f"Block {i:2d} | attn has_nan={has_nan_attn}, has_inf={has_inf_attn} | block has_nan={has_nan_block} | norm={hidden_states.norm().item():.2f}")
            if has_nan_block:
                print(f"NaN first appeared at Block {i}!")
                # Inspect q, k, v in this block
                print(f"q: min={q.min().item():.2f}, max={q.max().item():.2f}")
                print(f"k: min={k.min().item():.2f}, max={k.max().item():.2f}")
                print(f"v: min={v.min().item():.2f}, max={v.max().item():.2f}")
                break

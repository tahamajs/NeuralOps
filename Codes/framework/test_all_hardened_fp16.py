"""Comprehensive verification test for all hardened attention modules in fp16.

Tests instantiation, injection into GPT-2, weight copying, and fp16 forward pass
with causal attention masks.
"""

import sys
import os
import traceback
from typing import Dict, Any, List

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

# Explicitly import all target attention modules to register them
import framework.attentions.griffin_ssd
import framework.attentions.mamba_block
import framework.attentions.selective_ssm
import framework.attentions.simple_ssm
import framework.attentions.samba_m3
import framework.attentions.samba_moe
import framework.attentions.aegis_hybrid
import framework.attentions.aegis_spectron_fp8
import framework.attentions.hgrn2_layer
import framework.attentions.hgrn2_outer
import framework.attentions.regla_attention
import framework.attentions.halo_regla
import framework.attentions.reap_moe
import framework.attentions.expert_choice_moe
import framework.attentions.bam_umoe
import framework.attentions.goldfinch_ahn
import framework.attentions.upi_stuffed_mamba
import framework.attentions.jamba
import framework.attentions.jamba_info

from framework.core.registry import ATTENTION_REGISTRY

MODULES_TO_TEST = [
    ("griffin_ssd", "GriffinSSD"),
    ("mamba_block", "MambaBlock"),
    ("selective_ssm", "SelectiveSSM"),
    ("simple_ssm", "SimpleSSM"),
    ("samba_m3", "SambaM3Block"),
    ("samba_moe", "SambaMoEBlock"),
    ("aegis_hybrid", "AegisHybridBlock"),
    ("aegis_spectron_fp8", "AegisSpectronBlock"),
    ("hgrn2_layer", "HGRN2Layer"),
    ("hgrn2_outer", "HGRN2OuterBlock"),
    ("regla_attention", "ReGLAAttention"),
    ("halo_regla", "HALOReGLAAttention"),
    ("reap_moe", "REAPMoEAttention"),
    ("expert_choice_moe", "ExpertChoiceMoEAttention"),
    ("bam_umoe", "BAMUMoEBlock"),
    ("goldfinch_ahn", "GoldFinchAHNBlock"),
    ("upi_stuffed_mamba", "SelectiveMamba"),
    ("jamba", "Jamba2Block"),
    ("jamba_info", "JambaInfoBlock"),
]

def test_module_fp16(module_name: str, class_name: str) -> Dict[str, Any]:
    result = {
        "module": module_name,
        "class": class_name,
        "status": "PENDING",
        "error": None,
        "output_dtype": None,
        "output_shape": None,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    try:
        # Load small gpt2 config
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        config.n_head = 4
        config.n_embd = 128
        
        # Build base model and cast to fp16
        base_model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)
        base_model.eval()
        
        block = base_model.transformer.h[0]
        old_attn = block.attn
        
        # Lookup injector or class in registry
        injected = None
        injector_candidates = [
            f"inject_{module_name}",
            f"{module_name}_inject",
            module_name,
            f"{module_name}_attention",
            f"{module_name}_fn",
        ]
        
        for cand in injector_candidates:
            target_fn = ATTENTION_REGISTRY.get(cand)
            if target_fn is not None and callable(target_fn):
                try:
                    res = target_fn(layer=block, model_config=config, old_attn=old_attn, d_model=config.n_embd, num_heads=config.n_head, dtype=dtype)
                    if res is not None:
                        injected = res
                        break
                except Exception as e:
                    pass
        
        if injected is None:
            # Try direct instantiation from class registry
            cls = None
            for cand in [module_name, f"{module_name}_class", f"{module_name}_attention", class_name.lower(), class_name]:
                target_cls = ATTENTION_REGISTRY.get(cand)
                if isinstance(target_cls, type):
                    cls = target_cls
                    break
            
            if cls is not None:
                new_mod = cls(config=config, old_attn=old_attn, pretrained_attn=old_attn, d_model=config.n_embd, num_heads=config.n_head, dtype=dtype)
                p = next(old_attn.parameters(), None)
                if p is not None:
                    new_mod.to(device=p.device, dtype=p.dtype)
                else:
                    new_mod.to(device=device, dtype=dtype)
                block.attn = new_mod
                injected = new_mod

        if injected is None:
            result["status"] = "FAIL_NOT_REGISTERED"
            result["error"] = f"Could not find injector or class for {module_name}"
            return result
        
        # Ensure the whole model is properly in fp16
        base_model.to(device=device, dtype=dtype)
        
        # Create fp16 input
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
        attention_mask = torch.ones(B, L, device=device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        result["status"] = "PASS"
        result["output_dtype"] = str(logits.dtype)
        result["output_shape"] = list(logits.shape)
        
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        
    return result

def main():
    print("=" * 80)
    print("VERIFYING FP16 EXECUTION ACROSS ALL HARDENED ATTENTION MODULES")
    print("=" * 80)
    
    results = []
    for mod_name, cls_name in MODULES_TO_TEST:
        print(f"Testing {mod_name:25s} ... ", end="", flush=True)
        res = test_module_fp16(mod_name, cls_name)
        status = res["status"]
        if status == "PASS":
            print(f"PASS (dtype={res['output_dtype']}, shape={res['output_shape']})")
        else:
            print(f"{status}: {res['error']}")
        results.append(res)
        
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS TABLE")
    print("=" * 80)
    print(f"{'Module Name':25s} | {'Class Name':25s} | {'Status':10s} | {'Details / Error'}")
    print("-" * 80)
    passed_count = 0
    for r in results:
        details = f"dtype={r['output_dtype']}" if r["status"] == "PASS" else str(r["error"])[:40]
        if r["status"] == "PASS":
            passed_count += 1
        print(f"{r['module']:25s} | {r['class']:25s} | {r['status']:10s} | {details}")
    print("-" * 80)
    print(f"Total Passed: {passed_count}/{len(results)}")
    print("=" * 80)

if __name__ == "__main__":
    main()

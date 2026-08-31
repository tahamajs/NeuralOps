import sys
import time
from pathlib import Path
import copy

proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model
import framework.attentions
import framework.compression_engine
from framework.data.cached_tensors import get_cached_tokenized_dataset

# 5 Test Pairs to verify
TEST_PAIRS = [
    ("adaptive_span", "adaptive_sparse"),
    ("compressed_attention", "compressed_attention_layer_energy"),
    ("flash3", "flash3_fp8"),
    ("eat", "fusion_attn"),
    ("aegis_hybrid", "aegis_v2"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Phase 2 Verification on device: {device}")

# 1. Load pristine model template & pre-tokenized dataset
ckpt_candidates = [
    proj_dir / "models" / "distilbert_sst2",
    Path("D:/SVD_Project/models/distilbert_sst2"),
]
ckpt = None
for c in ckpt_candidates:
    if c.exists() and ((c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()):
        ckpt = str(c)
        break
if ckpt is None:
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Loading base checkpoint: {ckpt}")
base_template = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
pristine_sd = {k: v.cpu().clone() for k, v in base_template.state_dict().items()}

cached_splits = get_cached_tokenized_dataset("sst2", "distilbert", seq_len=128)
train_ds = cached_splits["train"]

results = {}

for pair_idx, (mod_a, mod_b) in enumerate(TEST_PAIRS, 1):
    print(f"\n============================================================")
    print(f"PAIR {pair_idx}: '{mod_a}' vs '{mod_b}'")
    print(f"============================================================")
    
    for mod_name in [mod_a, mod_b]:
        # Clean copy of base model
        model = copy.deepcopy(base_template)
        model.load_state_dict(pristine_sd)
        
        # Determine module category
        if mod_name in ATTENTION_REGISTRY:
            m_type = "attention"
        elif mod_name in COMPRESSOR_REGISTRY:
            m_type = "compressor"
        else:
            raise ValueError(f"Module {mod_name} not found in registries!")
            
        print(f"\n--- Testing Module: '{mod_name}' (type={m_type}) ---")
        
        # Inject module
        model, _ = apply_module_to_model(model, mod_name, m_type, rank=16, device=device)
        model.to(device)
        model.train()
        
        # CHECK 1: Injected layer Python type inspection
        # For DistilBERT, inspect transformer layer 0 attention
        layer0_attn = model.distilbert.transformer.layer[0].attention
        inner_attn = getattr(layer0_attn, "inner", layer0_attn)
        actual_type = type(inner_attn)
        print(f"Injected layer[0].attention type: {type(layer0_attn).__name__}")
        print(f"Underlying inner attention class: {actual_type.__module__}.{actual_type.__name__}")
        
        # Setup per-module seeded generator (matching run_quality_sweep.py)
        seed = 42
        _dl_seed = abs(hash(f"{mod_name}_{seed}")) % (2**31)
        _gen = torch.Generator()
        _gen.manual_seed(_dl_seed)
        
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
            generator=_gen,
        )
        
        # Discriminative optimizer setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Step training for 15 steps
        step_losses = []
        global_step = 0
        for b_ids, b_mask, b_lbl in loader:
            if global_step >= 15:
                break
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_lbl = b_lbl.to(device=device, dtype=torch.long, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
                loss = out.loss
                
            loss.backward()
            optimizer.step()
            
            step_losses.append(round(float(loss.item()), 4))
            global_step += 1
            
        results[mod_name] = {
            "inner_class": f"{actual_type.__module__}.{actual_type.__name__}",
            "losses": step_losses,
        }
        print(f"Step 1-15 Training Losses: {step_losses}")

print("\n" + "="*70)
print("PHASE 2 SUMMARY MATRIX & PAIRWISE COMPARISON")
print("="*70)
for pair_idx, (mod_a, mod_b) in enumerate(TEST_PAIRS, 1):
    res_a = results[mod_a]
    res_b = results[mod_b]
    class_distinct = (res_a["inner_class"] != res_b["inner_class"])
    loss_distinct = (res_a["losses"] != res_b["losses"])
    print(f"\nPair {pair_idx}: {mod_a} vs {mod_b}")
    print(f"  Class {mod_a}: {res_a['inner_class']}")
    print(f"  Class {mod_b}: {res_b['inner_class']}")
    print(f"  Distinct Classes: {class_distinct}")
    print(f"  Losses {mod_a}: {res_a['losses'][:10]}")
    print(f"  Losses {mod_b}: {res_b['losses'][:10]}")
    print(f"  Distinct Trajectories: {loss_distinct}")
    assert class_distinct, f"FAIL: Classes for {mod_a} and {mod_b} are identical ({res_a['inner_class']})!"
    assert loss_distinct, f"FAIL: Loss trajectories for {mod_a} and {mod_b} are identical!"

print("\n>>> ALL 5 TEST PAIRS PASSED OBJECT AND TRAJECTORY DIVERSITY VERIFICATION! <<<")

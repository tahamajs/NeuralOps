"""
investigate_issues_1_2_3.py
Thorough investigation of:
- Issue 1: Collapse audit raw logits & explanation of Class 0 vs Class 1 discrepancy
- Issue 2: Orion reproducibility across 3 runs with fixed seed
- Issue 3: Exact parameter counts for goldfinch_ahn, griffin_ssd, regla_attention
"""
import os, sys, csv, math, json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Codes" / "framework"))

def main():
    print("="*70)
    print("INVESTIGATION 1: COLLAPSE AUDIT LOGITS & CLASSIFIER BIAS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "models/distilbert_mnli_full"
    ds_dir = "Codes/framework/data/glue_data/mnli"
    sweep_dir = Path("results/distilbert_mnli_full_mnli_sweep")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ds = load_from_disk(ds_dir)["validation"]
    
    # 1. Check base model classifier head
    base_model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    ckpt = sweep_dir / "finetuned_model.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        base_model.load_state_dict(sd, strict=False)
    
    print("DistilBERT pre_classifier weight norm:", base_model.pre_classifier.weight.norm().item())
    print("DistilBERT classifier bias:", base_model.classifier.bias.data.tolist())
    zero_in = torch.zeros(1, 128, 768)
    with torch.no_grad():
        # DistilBERT pooled output is pre_classifier(x[:, 0]) -> relu -> dropout -> classifier
        pooled = torch.relu(base_model.pre_classifier(zero_in[:, 0]))
        zero_logits = base_model.classifier(pooled).tolist()[0]
    print("Logits for all-zero transformer output:", zero_logits)
    print("Argmax for all-zero output:", int(np.argmax(zero_logits)))
    
    # Test sample 0
    sample_text = f"{ds[0]['premise']} [SEP] {ds[0]['hypothesis']}"
    enc = tokenizer([sample_text], padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
    
    logits_dir = sweep_dir / "audit_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    
    modules_to_test = ["mamba", "simple_ssm", "mla_attention", "aegis_spectron_fp8", "galerkin"]
    from framework.eval_all_models_datasets import apply_module_to_model
    
    for mod in modules_to_test:
        print(f"\n--- Testing module: {mod} ---")
        m = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
        if ckpt.exists():
            m.load_state_dict(sd, strict=False)
        m = m.to(device)
        
        # Inject
        apply_module_to_model(m, mod, module_type="auto", device=device)
        m = m.half().eval()
        
        # Sample 0 logits with autocast
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                out = m(**enc)
            sample_logits = out.logits[0].float().cpu().numpy()
        print(f"Sample 0 raw logits for {mod}: {sample_logits.tolist()}")
        print(f"Sample 0 prediction (argmax): {int(np.argmax(sample_logits))}")
        
        # Compute mean logits over all 1000 validation samples
        all_logits = []
        batch_size = 64
        texts = [f"{r['premise']} [SEP] {r['hypothesis']}" for r in ds]
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i:i+batch_size]
                b_enc = tokenizer(b_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    b_out = m(**b_enc)
                all_logits.append(b_out.logits.float().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        np.save(logits_dir / f"{mod}.npy", all_logits)
        mean_vec = all_logits.mean(axis=0)
        std_vec = all_logits.std(axis=0)
        preds = np.argmax(all_logits, axis=1)
        print(f"Mean logit vector over N=1000: {mean_vec.tolist()}")
        print(f"Std of logits over N=1000:     {std_vec.tolist()}")
        print(f"Unique predictions count:       {dict(zip(*np.unique(preds, return_counts=True)))}")
        del m
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("INVESTIGATION 2: ORION REPRODUCIBILITY (3 RUNS, SEED=42)")
    print("="*70)
    from framework.run_quality_sweep import prepare_eval_batches, evaluate_comprehensive
    
    # Check dropout and eval settings
    eval_batches = prepare_eval_batches(tokenizer, ds, "mnli", 128, 32, device)
    
    orion_accs = []
    for run_idx in range(1, 4):
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)
        
        om = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
        if ckpt.exists():
            om.load_state_dict(sd, strict=False)
        om = om.to(device)
        apply_module_to_model(om, "orion", module_type="auto", device=device)
        om = om.half().eval()
        
        # Check if any dropout is training mode
        has_train_dropout = any(mod.training for mod in om.modules() if isinstance(mod, torch.nn.Dropout))
        
        perf = evaluate_comprehensive(om, tokenizer, ds, "mnli", 128, 32, device, fp16=True, precomputed_batches=eval_batches)
        acc = perf["accuracy_pct"]
        orion_accs.append(acc)
        print(f"Run {run_idx} (seed=42): acc={acc:.2f}%, F1={perf['f1_macro']:.4f}, training_dropout_present={has_train_dropout}")
        del om
        torch.cuda.empty_cache()
    
    print(f"Orion accuracies across 3 identical eval passes: {orion_accs}")
    reproducible = (len(set(orion_accs)) == 1)
    print(f"Eval is strictly deterministic with fixed seed: {reproducible}")

    print("\n" + "="*70)
    print("INVESTIGATION 3: CAT-B DELTA PARAMETER COUNTS")
    print("="*70)
    summary_path = sweep_dir / "summary.csv"
    with open(summary_path, encoding="utf-8") as f:
        rows = {r.get("module_name", r.get("module")): r for r in csv.DictReader(f)}
    
    cat_b_mods = ["goldfinch", "goldfinch_ahn", "griffin_ssd", "regla", "regla_attention"]
    print(f"{'Module':20} | {'Base Before':12} | {'Base After':12} | {'Delta Params':12} | {'Delta %':8} | {'LoRA Params':12}")
    print("-" * 85)
    for m in cat_b_mods:
        if m in rows:
            r = rows[m]
            p_before = int(r.get("total_params_before", 0) or 0)
            p_after = int(r.get("total_params_after", 0) or 0)
            lora = int(r.get("lora_params", 0) or 0)
            base_after = p_after - lora
            delta = base_after - p_before
            delta_pct = (delta / p_before) * 100.0 if p_before > 0 else 0.0
            print(f"{m:20} | {p_before:12,d} | {base_after:12,d} | {delta:12,d} | {delta_pct:7.2f}% | {lora:12,d}")

if __name__ == "__main__":
    main()

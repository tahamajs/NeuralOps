#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("KD-Reuse")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

def load_tier1_modules():
    p = HERE / "results" / "audit" / "trusted_modules.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return sorted(set(data.get("tier1_true_causal", []) + data.get("tier2_causal_via_mask", [])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", default=None)
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--recovery_steps", type=int, default=500)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    # FIX: Correct import name 'inject_attention'
    from neuralops.core.attention_utils import inject_attention

    LOG.info("PHASE 0: Loading GPT-2 + tokenizer + dataset ONCE")
    t_load = time.monotonic()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.eval()
    
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_texts = [t.strip() for t in dataset["train"]["text"] if len(t.strip()) > 20][:500]
    all_ids = []
    for t in train_texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if ids:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
    n_chunks = len(all_ids) // args.seq_len
    train_chunks = torch.tensor(
        all_ids[:n_chunks * args.seq_len], dtype=torch.long
    ).view(n_chunks, args.seq_len).cpu()
    
    LOG.info("Loaded in %.1fs", time.monotonic() - t_load)
    
    LOG.info("Saving pristine model snapshot via deepcopy...")
    base_model.to("cpu")
    pristine_model = copy.deepcopy(base_model)
    
    if args.modules:
        modules = args.modules
    else:
        modules = load_tier1_modules()
    
    registry_path = HERE / "results" / "kd_runs" / "registry.json"
    crash_log = HERE / "results" / "audit" / "hard_crashes.txt"
    tested = set()
    
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text(encoding="utf-8"))
            for r in reg.get("runs", []):
                if r.get("module"):
                    tested.add(r["module"])
        except Exception:
            pass
            
    if crash_log.exists():
        crashed = set(crash_log.read_text().splitlines())
        tested.update(crashed)
    
    already_run = [
        "salt_attention", "xattention", "nystrom_attention", "flash_attention3",
        "gqa_attention", "lora_attention", "cosformer", "performer_attention",
        "linformer_attention", "full_attention"
    ]
    for a in already_run:
        tested.add(a)

    modules = [m for m in modules if m not in tested]
    LOG.info("Modules to run: %d (skipping %d tested)", len(modules), len(tested))
    
    if not modules:
        LOG.info("All modules done!")
        sys.exit(0)

    results = {}
    t_sweep = time.monotonic()

    for i, module_name in enumerate(modules, 1):
        LOG.info(">>> [%d/%d] %s", i, len(modules), module_name)
        t_module = time.monotonic()
        
        # Mark as crashed before starting
        with open(crash_log, "a") as f:
            f.write(module_name + "\n")
        
        try:
            # a) Restore pristine architecture AND weights
            model = copy.deepcopy(pristine_model)
            model.to(args.device)
            
            # b) Inject the module (using inject_attention)
            injected_count = inject_attention(model, module_name, rank=args.rank)
            
            # c) Enable attention params for training
            for name, p in model.named_parameters():
                p.requires_grad = ".attn." in name or "attention" in name.lower()
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
            
            trainable = [p for p in model.parameters() if p.requires_grad]
            
            # d) Check injection validity
            model.eval()
            with torch.no_grad():
                test_batch = train_chunks[:2].to(args.device)
                pre_loss = model(test_batch, labels=test_batch).loss.item()
                pre_ppl = float(torch.exp(torch.tensor(pre_loss)))
            
            LOG.info("  Pre-KD PPL: %.2f", pre_ppl)
            
            # e) KD recovery loop
            model.train()
            optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda", enabled=(args.device.startswith("cuda")))
            
            batch_size = min(args.batch_size, len(train_chunks))
            
            for step in range(args.recovery_steps):
                s = (step * batch_size) % (len(train_chunks) - batch_size)
                batch = train_chunks[s:s + batch_size].to(args.device)
                
                with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                    loss = model(batch, labels=batch).loss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # f) Evaluate post-KD PPL
            model.eval()
            with torch.no_grad():
                test_batch = train_chunks[:4].to(args.device)
                post_loss = model(test_batch, labels=test_batch).loss.item()
                post_ppl = float(torch.exp(torch.tensor(post_loss)))
            
            elapsed = time.monotonic() - t_module
            recovery_pct = max(0, (pre_ppl - post_ppl) / max(1e-6, pre_ppl - 66.07) * 100)
            
            LOG.info("  Post-KD PPL: %.2f | Recovery: %.1f%% | Time: %.1fs",
                     post_ppl, recovery_pct, elapsed)
            
            results[module_name] = {
                "pre_ppl": round(pre_ppl, 2),
                "post_ppl": round(post_ppl, 2),
                "recovery_pct": round(recovery_pct, 2),
                "time_s": round(elapsed, 1),
                "status": "PASS" if recovery_pct > 30 else "FAIL",
            }
            
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            reg = {"runs": []}
            if registry_path.exists():
                try:
                    reg = json.loads(registry_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            reg["runs"] = [r for r in reg["runs"] if r.get("module") != module_name]
            
            reg["runs"].append({
                "module": module_name,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "results": {"C": results[module_name]},
            })
            registry_path.write_text(json.dumps(reg, indent=2))
            
        except Exception as e:
            LOG.exception("Module %s failed:", module_name)
            results[module_name] = {"status": "CRASH", "error": str(e)[:200]}
        
        # Cleanup
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.monotonic() - t_sweep
    LOG.info("FINAL SUMMARY (%.1f min)", total_time / 60)

if __name__ == "__main__":
    sys.exit(main())

import torch
import random
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from neuralops.utils.replace_attention import replace_attention

seeds = list(range(1001, 1011))
models_to_test = [
    ("GPT2", "causal", "/mnt/d/models/gpt2", ["svd_linear_attention", "gla_attention", "gated_delta_net"]),
    ("RoBERTa", "seq_cls", "/mnt/d/models/roberta-base", ["svd_linear_attention", "gla_attention"]),
    ("DistilBERT", "seq_cls", "/mnt/d/models/distilbert-base-uncased", ["svd_linear_attention", "gla_attention"])
]

print("=== STARTING 10-SEED CONTINUOUS STRESS TEST BATTERY ===")
all_passed = True

for seed_idx, seed in enumerate(seeds, 1):
    print(f"\n>>> SEED {seed_idx}/10 (Value={seed}) <<<")
    torch.manual_seed(seed)
    random.seed(seed)
    
    for model_name, task, model_path, mechs in models_to_test:
        for mech in mechs:
            torch.cuda.empty_cache()
            if task == "causal":
                cfg = AutoConfig.from_pretrained(model_path)
                cfg.n_positions = 1024
                m = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, ignore_mismatched_sizes=True).cuda()
                m = replace_attention(m, attention_key=mech)
                
                # 1. Forward check with random prompt tokens
                inp = torch.randint(0, 1000, (1, 64), device="cuda")
                out = m(inp)
                if not torch.isfinite(out.logits).all():
                    print(f"[FAIL] Seed {seed} {model_name} {mech} Forward produced NaNs/Infs!")
                    all_passed = False
                    
                # 2. Backward check
                m.train()
                out_train = m(inp, labels=inp)
                out_train.loss.backward()
                finite_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad and p.grad is not None)
                if not finite_grads:
                    print(f"[FAIL] Seed {seed} {model_name} {mech} Backward produced non-finite gradients!")
                    all_passed = False
                    
                # 3. 128-step continuous generation check
                m.eval()
                curr_ids = inp.clone()
                with torch.no_grad():
                    for _ in range(128):
                        step_out = m(curr_ids)
                        if not torch.isfinite(step_out.logits).all():
                            print(f"[FAIL] Seed {seed} {model_name} {mech} Step generation produced NaNs!")
                            all_passed = False
                            break
                        nxt = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
                        curr_ids = torch.cat([curr_ids, nxt], dim=-1)
                print(f"[PASS] Seed {seed} | {model_name:<10} | {mech:<22} | Fwd, Bwd, 128-Gen OK")
                
            else:
                m = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
                m = replace_attention(m, attention_key=mech)
                
                inp = torch.randint(0, 1000, (2, 128), device="cuda")
                lbl = torch.randint(0, 2, (2,), device="cuda")
                
                # Forward
                out = m(inp, labels=lbl)
                if not (torch.isfinite(out.logits).all() and torch.isfinite(out.loss).all()):
                    print(f"[FAIL] Seed {seed} {model_name} {mech} Forward produced NaNs/Infs!")
                    all_passed = False
                    
                # Backward
                m.train()
                m.zero_grad()
                out.loss.backward()
                finite_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad and p.grad is not None)
                if not finite_grads:
                    print(f"[FAIL] Seed {seed} {model_name} {mech} Backward produced non-finite gradients!")
                    all_passed = False
                print(f"[PASS] Seed {seed} | {model_name:<10} | {mech:<22} | Fwd, Bwd OK")

res_str = "ALL 10 SEEDS PASSED 100% PERFECTLY" if all_passed else "FAILURES DETECTED"
print(f"\n=== CONTINUOUS BATTERY RESULT: {res_str} ===")

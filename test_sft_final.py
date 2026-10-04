import torch
import sys
import time
sys.path.insert(0, '.')

import neuralops.benchmarks.run_ruler_benchmark as ruler_mod
ruler_mod.torch = torch
ruler_mod.np = __import__('numpy')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.benchmarks.run_ruler_benchmark import evaluate_context_length

MODULE = 'gqa_attention'
STAGE1_STEPS = 50
STAGE2_STEPS = 500
SEQ_LEN = 128
BATCH = 4
LR1 = 5e-4
LR2 = 1e-4

print(f'Loading + injecting {MODULE}...')
tok = AutoTokenizer.from_pretrained('gpt2')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model, stats = apply_module_to_model(model, MODULE, 'attention', rank=64, device='cuda', tokenizer=tok)
model.cuda()

ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
texts = [t.strip() for t in ds['train']['text'] if len(t.strip()) > 20][:500]
all_ids = []
for t in texts:
    ids = tok(t, add_special_tokens=False).input_ids
    if ids:
        all_ids.extend(ids)
        all_ids.append(tok.eos_token_id)
n = len(all_ids) // SEQ_LEN
chunks = torch.tensor(all_ids[:n*SEQ_LEN], dtype=torch.long).view(n, SEQ_LEN)
print(f'Chunks: {chunks.shape}')


def niah_eval(model, tok, label):
    print(f'NIAH {label}...')
    model.eval()
    try:
        r = evaluate_context_length(
            model=model, tokenizer=tok, context_len=512,
            depths=[0.1, 0.5, 0.9], num_samples=4,
            max_position=1024, device=torch.device('cuda'),
        )
        print(f'  Result: {r["accuracy"]:.2%}')
        return r['accuracy']
    except Exception as e:
        print(f'  FAILED: {e}')
        return 0.0


acc0 = niah_eval(model, tok, 'BEFORE')

print(f'\n=== STAGE 1 ({STAGE1_STEPS} steps/layer, LR={LR1}) ===')
for li in range(len(model.transformer.h)):
    for p in model.parameters():
        p.requires_grad = False
    tr = []
    for name, p in model.named_parameters():
        if f'transformer.h.{li}.' in name and '.attn.' in name:
            p.requires_grad = True
            if p.dtype != torch.float32:
                p.data = p.data.float()
            tr.append(p)
    if not tr:
        continue
    opt = torch.optim.AdamW(tr, lr=LR1)
    sc = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    losses = []
    for step in range(STAGE1_STEPS):
        s = (li * 100 + step * BATCH) % (len(chunks) - BATCH)
        b = chunks[s:s+BATCH].cuda()
        with torch.amp.autocast('cuda'):
            loss = model(b, labels=b).loss
        opt.zero_grad()
        sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(tr, 1.0)
        sc.step(opt)
        sc.update()
        losses.append(loss.item())
    print(f'  L{li:2d}: {losses[0]:.3f} -> {losses[-1]:.3f}')

acc1 = niah_eval(model, tok, 'AFTER S1')

print(f'\n=== STAGE 2 ({STAGE2_STEPS} steps, LR={LR2}) ===')
tr = []
for name, p in model.named_parameters():
    if '.attn.' in name:
        p.requires_grad = True
        if p.dtype != torch.float32:
            p.data = p.data.float()
        tr.append(p)
    else:
        p.requires_grad = False
print(f'Joint trainable: {sum(p.numel() for p in tr):,}')

opt = torch.optim.AdamW(tr, lr=LR2, weight_decay=0.01)
sc = torch.cuda.amp.GradScaler(enabled=True)
model.train()
t0 = time.monotonic()
for step in range(STAGE2_STEPS):
    s = (step * BATCH) % (len(chunks) - BATCH)
    b = chunks[s:s+BATCH].cuda()
    with torch.amp.autocast('cuda'):
        loss = model(b, labels=b).loss
    opt.zero_grad()
    sc.scale(loss).backward()
    sc.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(tr, 1.0)
    sc.step(opt)
    sc.update()
    if (step+1) % 100 == 0 or step == 0:
        print(f'  Step {step+1}/{STAGE2_STEPS} | Loss: {loss.item():.4f}')
print(f'Stage 2 done in {time.monotonic()-t0:.1f}s')

acc2 = niah_eval(model, tok, 'AFTER S2')

print()
print('=' * 60)
print(f'NIAH before:     {acc0:.2%}')
print(f'NIAH after S1:   {acc1:.2%}')
print(f'NIAH after S2:   {acc2:.2%}')
print('=' * 60)

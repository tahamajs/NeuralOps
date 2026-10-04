import torch
import sys
import time
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.benchmarks.run_ruler_benchmark import evaluate_context_length

MODULE = 'gqa_attention'
STAGE1_STEPS_PER_LAYER = 20
STAGE2_STEPS = 200
SEQ_LEN = 128
BATCH_SIZE = 2
LR_STAGE1 = 1e-5
LR_STAGE2 = 5e-5

print(f'=== Loading GPT-2 + injecting {MODULE} ===')
tok = AutoTokenizer.from_pretrained('gpt2')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model, stats = apply_module_to_model(model, MODULE, 'attention', rank=64, device='cuda', tokenizer=tok)
model.cuda()
print('Injection stats:', stats)

# Training data
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
texts = [t.strip() for t in ds['train']['text'] if len(t.strip()) > 20][:300]
all_ids = []
for t in texts:
    ids = tok(t, add_special_tokens=False).input_ids
    if ids:
        all_ids.extend(ids)
        all_ids.append(tok.eos_token_id)
n_chunks = len(all_ids) // SEQ_LEN
chunks = torch.tensor(all_ids[:n_chunks * SEQ_LEN], dtype=torch.long).view(n_chunks, SEQ_LEN)
print(f'Training chunks: {chunks.shape}')

# Quick sanity forward
print('\n=== Sanity forward ===')
model.eval()
with torch.no_grad():
    test_batch = chunks[:2].cuda()
    out = model(test_batch, labels=test_batch)
    print(f'Forward OK, loss={out.loss.item():.4f}')
    # Quick generation test
    ids = tok.encode('The secret is 42. Question: What is the secret? Answer:', return_tensors='pt').cuda()
    gen = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f'Generation test: {repr(tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))}')

def niah_eval(model, tok, label):
    print(f'\n=== NIAH {label} ===')
    model.eval()
    try:
        r = evaluate_context_length(
            model=model, tokenizer=tok, context_len=512,
            depths=[0.1, 0.5, 0.9], num_samples=4,
            max_position=1024, device=torch.device('cuda'),
        )
        acc = r['accuracy']
        print(f'NIAH {label}: {acc:.2%}')
        return acc
    except Exception as e:
        import traceback
        print(f'NIAH {label} FAILED:')
        traceback.print_exc()
        return 0.0

acc_before = niah_eval(model, tok, 'BEFORE')

# STAGE 1: Layer-by-layer
print(f'\n=== STAGE 1: Layer-by-Layer ({STAGE1_STEPS_PER_LAYER} steps/layer) ===')
num_layers = len(model.transformer.h)
for layer_idx in range(num_layers):
    for p in model.parameters():
        p.requires_grad = False
    layer_trainable = []
    for name, p in model.named_parameters():
        if f'transformer.h.{layer_idx}.' in name and ('.attn.' in name or 'attention' in name.lower()):
            p.requires_grad = True
            if p.dtype != torch.float32:
                p.data = p.data.float()
            layer_trainable.append(p)
    if not layer_trainable:
        continue
    opt = torch.optim.AdamW(layer_trainable, lr=LR_STAGE1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    losses = []
    for step in range(STAGE1_STEPS_PER_LAYER):
        s = (layer_idx * 50 + step * BATCH_SIZE) % (len(chunks) - BATCH_SIZE)
        batch = chunks[s:s+BATCH_SIZE].cuda()
        with torch.amp.autocast('cuda'):
            loss = model(batch, labels=batch).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(layer_trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    print(f'  Layer {layer_idx:2d}: loss {losses[0]:.3f} -> {losses[-1]:.3f}')

acc_stage1 = niah_eval(model, tok, 'AFTER STAGE 1')

# STAGE 2: Joint
print(f'\n=== STAGE 2: Joint ({STAGE2_STEPS} steps) ===')
trainable = []
for name, p in model.named_parameters():
    if '.attn.' in name or 'attention' in name.lower():
        p.requires_grad = True
        if p.dtype != torch.float32:
            p.data = p.data.float()
        trainable.append(p)
    else:
        p.requires_grad = False
print(f'Joint trainable: {sum(p.numel() for p in trainable):,} params')

opt = torch.optim.AdamW(trainable, lr=LR_STAGE2, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=True)
model.train()
t0 = time.monotonic()
for step in range(STAGE2_STEPS):
    s = (step * BATCH_SIZE) % (len(chunks) - BATCH_SIZE)
    batch = chunks[s:s+BATCH_SIZE].cuda()
    with torch.amp.autocast('cuda'):
        loss = model(batch, labels=batch).loss
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(opt)
    scaler.update()
    if (step+1) % 50 == 0 or step == 0:
        print(f'  Step {step+1}/{STAGE2_STEPS} | Loss: {loss.item():.4f}')
print(f'Stage 2 done in {time.monotonic()-t0:.1f}s')

acc_stage2 = niah_eval(model, tok, 'AFTER STAGE 2')

print()
print('=' * 65)
print(f'MODULE: {MODULE}')
print(f'NIAH before SFT:       {acc_before:.2%}')
print(f'NIAH after Stage 1:    {acc_stage1:.2%}')
print(f'NIAH after Stage 2:    {acc_stage2:.2%}')
print('=' * 65)

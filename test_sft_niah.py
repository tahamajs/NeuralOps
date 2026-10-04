import torch
import sys
import time
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.benchmarks.run_ruler_benchmark import evaluate_context_length

MODULE = 'gqa_attention'
SFT_STEPS = 200
SEQ_LEN = 128
BATCH_SIZE = 2
LR = 5e-5

print(f'=== Loading GPT-2 + injecting {MODULE} ===')
tok = AutoTokenizer.from_pretrained('gpt2')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model, stats = apply_module_to_model(model, MODULE, 'attention', rank=64, device='cuda', tokenizer=tok)
model.cuda()
print('Injection stats:', stats)

# --- NIAH BEFORE SFT ---
print('\n=== NIAH BEFORE SFT ===')
model.eval()
try:
    r_before = evaluate_context_length(
        model=model, tokenizer=tok, context_len=512,
        depths=[0.1, 0.5, 0.9], num_samples=4,
        max_position=1024, device=torch.device('cuda'),
    )
    acc_before = r_before['accuracy']
    print(f'NIAH before SFT: {acc_before:.2%}')
except Exception as e:
    print(f'Failed: {e}')
    acc_before = 0.0

# --- SFT ---
print(f'\n=== SFT ({SFT_STEPS} steps) ===')
for name, p in model.named_parameters():
    p.requires_grad = '.attn.' in name or 'attention' in name.lower()
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()

trainable = [p for p in model.parameters() if p.requires_grad]
print(f'Trainable: {sum(p.numel() for p in trainable):,} params')
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)

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

model.train()
scaler = torch.cuda.amp.GradScaler(enabled=True)
t0 = time.monotonic()
for step in range(SFT_STEPS):
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
        print(f'  Step {step+1}/{SFT_STEPS} | Loss: {loss.item():.4f}')
print(f'SFT done in {time.monotonic()-t0:.1f}s')

# --- NIAH AFTER SFT ---
print('\n=== NIAH AFTER SFT ===')
model.eval()
try:
    r_after = evaluate_context_length(
        model=model, tokenizer=tok, context_len=512,
        depths=[0.1, 0.5, 0.9], num_samples=4,
        max_position=1024, device=torch.device('cuda'),
    )
    acc_after = r_after['accuracy']
    print(f'NIAH after SFT: {acc_after:.2%}')
except Exception as e:
    print(f'Failed: {e}')
    acc_after = 0.0

print()
print('=' * 60)
print(f'MODULE: {MODULE}')
print(f'NIAH before SFT: {acc_before:.2%}')
print(f'NIAH after  SFT: {acc_after:.2%}')
print(f'Improvement:     {(acc_after - acc_before):+.2%}')
print('=' * 60)

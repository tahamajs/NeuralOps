import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from neuralops.attentions.hgrn2_gated_deltanet import inject_hgrn2_gated_deltanet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

inject_hgrn2_gated_deltanet(model)
model.to(device).eval()

train_like = 'The quick brown fox jumps over the lazy dog. ' * 20
test_like = 'Quantum chromodynamics predicts asymptotic freedom in high energy scattering. ' * 20
random_ids = torch.randint(0, 50257, (1, 128)).to(device)

inputs_train = tokenizer(train_like, return_tensors='pt', max_length=128, truncation=True).to(device)
inputs_test = tokenizer(test_like, return_tensors='pt', max_length=128, truncation=True).to(device)

with torch.no_grad():
    loss_train = model(**inputs_train, labels=inputs_train['input_ids']).loss.item()
    loss_test = model(**inputs_test, labels=inputs_test['input_ids']).loss.item()
    loss_rand = model(random_ids, labels=random_ids).loss.item()

ppl_train = math.exp(loss_train)
ppl_test = math.exp(loss_test)
ppl_rand = math.exp(loss_rand)

print(f"Train-like PPL: {ppl_train:.2f}")
print(f"Test-like PPL:  {ppl_test:.2f}")
print(f"Random PPL:     {ppl_rand:.2f}")
print("Vocab size:     50257")
print()

if ppl_rand < 1000:
    print("⚠️ LEAKAGE detected: random PPL too low")
elif ppl_train < ppl_test / 5:
    print("⚠️ OVERFITTING detected: train << test")
else:
    print("✅ Looks OK")

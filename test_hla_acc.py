import torch
import sys
sys.path.insert(0, 'Codes')
from transformers import RobertaForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention

m = RobertaForSequenceClassification.from_pretrained('models/roberta-base')
m.eval()
print("Baseline model loaded.")

# Inspect HLA
inject_attention(m, 'hla', rank=32)
m.eval()
print("HLA injected.")

# Load small eval batch
batch = torch.load('results/roberta_sst2_lora_baseline/eval_batch.pt') if False else None

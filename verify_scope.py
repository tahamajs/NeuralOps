from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model
from kd_quick_test import set_train_scope

m = AutoModelForCausalLM.from_pretrained('models/gpt2_wikitext')
m, _ = apply_module_to_model(m, 'mamba_block', 'attention', rank=64, device='cpu')
n = set_train_scope(m, 'mamba')
print(f'Trainable: {n:,}')

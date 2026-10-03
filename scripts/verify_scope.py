from transformers import AutoModelForCausalLM
from framework.eval_all_models_datasets import apply_module_to_model
from kd_quick_test import set_train_scope

for mod in ['mamba_block', 'aegis_hybrid']:
    m = AutoModelForCausalLM.from_pretrained('models/gpt2_wikitext')
    m, _ = apply_module_to_model(m, mod, 'attention', rank=64, device='cpu')
    n = set_train_scope(m, 'attention')
    print(f'{mod}: Trainable = {n:,}')
    assert n > 0, f'{mod} returned 0 trainable params!'
    del m

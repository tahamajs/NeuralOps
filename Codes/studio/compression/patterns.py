# studio/compression/patterns.py
ARCH_PATTERNS = {
    'bert': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'],
    'distilbert': ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin'],
    'roberta': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'],
    'albert': ['attention_query', 'attention_key', 'attention_value', 'attention_output'],
}

def detect_architecture(model_name: str) -> str:
    name = model_name.lower()
    if 'distilbert' in name: return 'distilbert'
    if 'roberta' in name: return 'roberta'
    if 'albert' in name: return 'albert'
    return 'bert'

def get_layer_patterns(arch: str):
    return ARCH_PATTERNS.get(arch, ARCH_PATTERNS['bert'])
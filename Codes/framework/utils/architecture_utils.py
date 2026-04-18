# Architecture patterns for linear layer compression
ARCH_PATTERNS = {
    'bert': [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense'
    ],
    'distilbert': [
        'attention.q_lin',
        'attention.k_lin',
        'attention.v_lin',
        'attention.out_lin'
    ],
    'roberta': [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense'
    ],
    'xlnet': [
        'layer.layer.0.attention.hybrid_attn.q_linear',
        'layer.layer.0.attention.hybrid_attn.k_linear',
        'layer.layer.0.attention.hybrid_attn.v_linear',
        'layer.layer.0.attention.output.dense'
    ],
    'albert': [
        'attention_query',
        'attention_key',
        'attention_value',
        'attention_output'
    ]
}

def detect_architecture(model_name: str) -> str:
    name_lower = model_name.lower()
    if 'distilbert' in name_lower: return 'distilbert'
    if 'roberta' in name_lower: return 'roberta'
    if 'xlnet' in name_lower: return 'xlnet'
    if 'albert' in name_lower: return 'albert'
    if 'bert' in name_lower: return 'bert'
    return 'bert'

def get_layer_patterns(arch: str) -> list:
    return ARCH_PATTERNS.get(arch, ARCH_PATTERNS['bert'])
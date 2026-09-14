import sys
sys.path.insert(0, 'Codes')
from framework.core.registry import ATTENTION_REGISTRY
import framework.attentions.svd_linear_attention
cls = ATTENTION_REGISTRY.get('svd_linear_attention')
print('Registered type for svd_linear_attention:', cls)

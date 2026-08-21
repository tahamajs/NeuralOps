import sys
from framework.tests.conftest import build_attention, model_config
from transformers import DistilBertConfig

config = DistilBertConfig(hidden_size=768, num_attention_heads=12)
try:
    attn = build_attention("token_compressed_module", config)
    print("Success!", attn)
except Exception as e:
    print("Error:", type(e), e)

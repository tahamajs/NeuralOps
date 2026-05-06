from .base import BaseLoader, LOADER_REGISTRY
from .hf_loader import HFStandardLoader
from .unsloth_loader import UnslothLoader
from .safetensors_loader import SafeTensorsLoader
from .flash_attn_loader import FlashAttentionLoader

def list_loaders():
    return list(LOADER_REGISTRY.keys())
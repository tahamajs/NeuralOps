from .base import LOADER_REGISTRY, BaseLoader
from .flash_attn_loader import FlashAttentionLoader
from .hf_loader import HFStandardLoader
from .safetensors_loader import SafeTensorsLoader
from .unsloth_loader import UnslothLoader


def list_loaders():
    return list(LOADER_REGISTRY.keys())

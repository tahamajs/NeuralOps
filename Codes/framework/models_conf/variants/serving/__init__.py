from .base import SERVER_REGISTRY, BaseServer
from .ray_serve import RayServeServer
from .tgi import TGIModelServer
from .triton import TritonServer
from .vllm import VLLMServer


def list_servers():
    return list(SERVER_REGISTRY.keys())

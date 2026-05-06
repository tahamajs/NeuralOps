from .base import BaseServer, SERVER_REGISTRY
from .vllm import VLLMServer
from .triton import TritonServer
from .tgi import TGIModelServer
from .ray_serve import RayServeServer

def list_servers():
    return list(SERVER_REGISTRY.keys())
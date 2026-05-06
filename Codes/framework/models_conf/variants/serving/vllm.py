from .base import BaseServer, SERVER_REGISTRY

@SERVER_REGISTRY.register("vllm")
class VLLMServer(BaseServer):
    def serve(self, model, **kwargs):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Install vllm for vLLM serving")
        llm = LLM(model=model)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # In practice, you'd expose an API endpoint; here we just return the engine
        return llm, sampling_params
from .base import SERVER_REGISTRY, BaseServer


@SERVER_REGISTRY.register("tgi")
class TGIModelServer(BaseServer):
    def serve(self, model, **kwargs) -> Any:
        """Serve."""
        try:
            from text_generation import Client
        except ImportError:
            raise ImportError("Install text-generation for HuggingFace TGI")
        # Placeholder – TGI is typically deployed as a separate container
        return "TGI client ready – point to a running TGI server"

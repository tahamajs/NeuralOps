"""Base class for compression methods."""

class BaseCompressor:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.original_params = sum(p.numel() for p in model.parameters())

    def compress(self):
        raise NotImplementedError

    def get_stats(self):
        compressed = sum(p.numel() for p in self.model.parameters())
        return {
            "original": self.original_params,
            "compressed": compressed,
            "ratio": compressed / self.original_params
        }

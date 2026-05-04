from ...utils import replace_attention   # adjust import path as needed
from framework.attentions.core.registry import AUX_REGISTRY   # <-- added

@AUX_REGISTRY.register("rala_compressor")   # <-- added
class RALACompressor:
    """
    Replaces every attention layer with the 'rala' variant (Rank‑Allocated Low‑rank Attention).
    """
    def __init__(self, model, tokenizer, rank: int = 8, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.device = device

    def compress(self, output_dir: str = None):
        self.model = replace_attention(self.model, "rala")
        if output_dir:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        return self.model, self.tokenizer
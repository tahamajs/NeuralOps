from framework.attentions.utils import replace_attention   # adjust import path as needed

class NHACompressor:
    """
    Replaces every attention layer with the 'nha' variant (Native Hybrid Attention).
    The replacement is performed via your framework's replace_attention utility.
    """
    def __init__(self, model, tokenizer, full_ratio: float = 0.1, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.full_ratio = full_ratio
        self.device = device

    def compress(self, output_dir: str = None):
        self.model = replace_attention(self.model, "nha")
        if output_dir:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        return self.model, self.tokenizer
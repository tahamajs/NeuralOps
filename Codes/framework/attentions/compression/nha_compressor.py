from framework.attentions.utils import replace_attention

class NHACompressor:
    def __init__(self, model, tokenizer, full_ratio=0.1, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.full_ratio = full_ratio
        self.device = device

    def compress(self, output_dir=None):
        self.model = replace_attention(self.model, "nha")
        if output_dir:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        return self.model, self.tokenizer
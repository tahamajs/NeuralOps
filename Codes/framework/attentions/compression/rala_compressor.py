from framework.attentions.utils import replace_attention

class RALACompressor:
    def __init__(self, model, tokenizer, rank=8, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.device = device

    def compress(self, output_dir=None):
        self.model = replace_attention(self.model, "rala")
        if output_dir:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        return self.model, self.tokenizer
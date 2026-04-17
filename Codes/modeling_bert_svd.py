from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertAttention
from svd_attention import SVDLinearAttention

class BertSVDAttention(BertAttention):
    def __init__(self, config, rank=64):
        super().__init__(config)
        self.self = SVDLinearAttention(config, rank=rank)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

class BertModelSVD(BertModel):
    def __init__(self, config, rank=64):
        super().__init__(config)
        for layer in self.encoder.layer:
            layer.attention = BertSVDAttention(config, rank=rank)
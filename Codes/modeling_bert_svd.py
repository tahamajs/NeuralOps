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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # Ensure output_attentions is taken from config if not explicitly set
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call parent forward with all arguments
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs
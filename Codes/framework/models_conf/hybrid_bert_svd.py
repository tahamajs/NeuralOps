import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
# from ..compression.hybrid.hybrid_svd_layer import DirectSVDEncoderLayer
class HybridBertWithDirectSVDBert(BertModel):
    """BERT model where every encoder layer uses DirectSVDEncoderLayer (no attention)."""
    def __init__(self, config):
        super().__init__(config)
        # Replace encoder layers
        self.encoder = nn.ModuleList([
            DirectSVDEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # Call parent embeddings (bypassing the parent's encoder)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = None
        if hasattr(self, 'pooler') and self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
            pooled_output = torch.tanh(pooled_output)

        # Return a simple tuple (compatible with BaseModelOutput)
        return (hidden_states, pooled_output, all_hidden_states)

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        return self.dense(hidden_states)

class HybridBertForSequenceClassification(nn.Module):
    """Wrapper that uses HybridBertWithDirectSVDBert + classification head."""
    def __init__(self, config):
        super().__init__()
        self.bert = HybridBertWithDirectSVDBert(config)
        self.classifier = ClassificationHead(config)
        self.num_labels = config.num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]  # pooler_output
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}
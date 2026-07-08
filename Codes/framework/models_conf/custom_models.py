"""
Experimental compressed BERT architectures (used only with variant="compressed" or "hybrid_svd").
These are not model‑agnostic and are kept for research purposes.
"""

import torch
import torch.nn as nn
from transformers import (AutoModelForSequenceClassification, BertConfig,
                          BertModel)


# ----------------------------------------------------------------------
# CompressedBertModel – SVD on embeddings, simple TransformerEncoder layers
# ----------------------------------------------------------------------
class CompressedBertModel(nn.Module):
    def __init__(self, model_path: str, rank: int, num_labels: int):
        super().__init__()
        config = BertConfig.from_pretrained(model_path)
        self.hidden_size = config.hidden_size
        self.rank = rank
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        base_bert = base_model.bert

        # Word embeddings (compressed via SVD)
        W_word = base_bert.embeddings.word_embeddings.weight.data.clone()
        U_w, S_w, Vh_w = torch.linalg.svd(W_word.float(), full_matrices=False)
        W_compressed_word = U_w[:, :rank] * S_w[:rank].unsqueeze(0)
        self.word_embeddings = nn.Embedding(
            base_bert.embeddings.word_embeddings.num_embeddings, rank
        )
        with torch.no_grad():
            self.word_embeddings.weight.copy_(W_compressed_word)

        # Position embeddings (compressed)
        W_pos = base_bert.embeddings.position_embeddings.weight.data.clone()
        U_p, S_p, Vh_p = torch.linalg.svd(W_pos.float(), full_matrices=False)
        W_compressed_pos = U_p[:, :rank] * S_p[:rank].unsqueeze(0)
        self.position_embeddings = nn.Embedding(
            base_bert.embeddings.position_embeddings.num_embeddings, rank
        )
        with torch.no_grad():
            self.position_embeddings.weight.copy_(W_compressed_pos)

        # Token type embeddings (optional, compressed)
        if hasattr(base_bert.embeddings, "token_type_embeddings"):
            W_tt = base_bert.embeddings.token_type_embeddings.weight.data.clone()
            actual_rank = min(rank, W_tt.shape[0])
            U_tt, S_tt, Vh_tt = torch.linalg.svd(W_tt.float(), full_matrices=False)
            W_compressed_tt = U_tt[:, :actual_rank] * S_tt[:actual_rank].unsqueeze(0)
            self.token_type_embeddings = nn.Embedding(
                base_bert.embeddings.token_type_embeddings.num_embeddings, actual_rank
            )
            with torch.no_grad():
                self.token_type_embeddings.weight.copy_(W_compressed_tt)
            self.tt_rank = actual_rank
        else:
            self.token_type_embeddings = None

        # Simplified encoder (TransformerEncoderLayer)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=rank, nhead=config.num_attention_heads, batch_first=True
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(rank)
        self.classifier = nn.Linear(rank, num_labels)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        word_embeds = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = word_embeds + position_embeds
        if self.token_type_embeddings is not None:
            token_type_ids = torch.zeros_like(input_ids)
            token_embeds = self.token_type_embeddings(token_type_ids)
            embeddings += token_embeds

        # Pass through encoder layers
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, src_key_padding_mask=~attention_mask.bool())
        embeddings = self.final_layer_norm(embeddings)
        cls_output = embeddings[:, 0, :]  # take the first token
        logits = self.classifier(cls_output)
        return {"logits": logits}


# ----------------------------------------------------------------------
# HybridBertForSequenceClassification – replaces BERT layers with DirectSVDEncoderLayer
# NOTE: The class `DirectSVDEncoderLayer` must be imported from wherever it is defined.
# We provide a placeholder. Replace with the actual import.
# ----------------------------------------------------------------------
class DirectSVDEncoderLayer(nn.Module):
    """Placeholder – replace with your actual implementation."""

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        return (self.linear(hidden_states),)


class HybridBertWithDirectSVDBert(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.ModuleList(
            [DirectSVDEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Use parent embeddings but our own encoder
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = kwargs["inputs_embeds"].size()[:-1]
        seq_length = input_shape[1]
        device = (
            input_ids.device
            if input_ids is not None
            else kwargs["inputs_embeds"].device
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=kwargs.get("position_ids"),
            token_type_ids=kwargs.get("token_type_ids"),
            inputs_embeds=kwargs.get("inputs_embeds"),
        )

        hidden_states = embedding_output
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)[0]

        pooled_output = self.pooler(hidden_states) if hasattr(self, "pooler") else None
        return (hidden_states, pooled_output)


class HybridBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = HybridBertWithDirectSVDBert(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.bert.config.num_labels), labels.view(-1)
            )
        return {"loss": loss, "logits": logits}

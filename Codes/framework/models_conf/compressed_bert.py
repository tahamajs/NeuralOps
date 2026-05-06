import torch
import torch.nn as nn
from transformers import BertConfig, AutoModelForSequenceClassification

class CompressedBertModel(nn.Module):
    def __init__(self, model_path: str, rank: int, num_labels: int):
        super().__init__()
        config = BertConfig.from_pretrained(model_path)
        self.config = config
        self.hidden_size = config.hidden_size
        self.rank = rank
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2) 
        base_bert = base_model.bert
        
        # Word Embeddings
        W_word = base_bert.embeddings.word_embeddings.weight.data.clone()
        U_w, S_w, Vh_w = torch.linalg.svd(W_word, full_matrices=False)
        W_compressed_word = U_w[:, :rank] * S_w[:rank].unsqueeze(0)
        self.word_embeddings = nn.Embedding(base_bert.embeddings.word_embeddings.num_embeddings, rank)
        with torch.no_grad():
            self.word_embeddings.weight.copy_(W_compressed_word)
        
        # Position Embeddings
        W_pos = base_bert.embeddings.position_embeddings.weight.data.clone()
        U_p, S_p, Vh_p = torch.linalg.svd(W_pos, full_matrices=False)
        W_compressed_pos = U_p[:, :rank] * S_p[:rank].unsqueeze(0)
        self.position_embeddings = nn.Embedding(base_bert.embeddings.position_embeddings.num_embeddings, rank)
        with torch.no_grad():
            self.position_embeddings.weight.copy_(W_compressed_pos)
        
        # Token Type Embeddings (optional)
        if hasattr(base_bert.embeddings, 'token_type_embeddings'):
            W_tt = base_bert.embeddings.token_type_embeddings.weight.data.clone()
            actual_rank = min(rank, W_tt.shape[0])
            U_tt, S_tt, Vh_tt = torch.linalg.svd(W_tt, full_matrices=False)
            W_compressed_tt = U_tt[:, :actual_rank] * S_tt[:actual_rank].unsqueeze(0)
            self.token_type_embeddings = nn.Embedding(base_bert.embeddings.token_type_embeddings.num_embeddings, actual_rank)
            with torch.no_grad():
                self.token_type_embeddings.weight.copy_(W_compressed_tt)
            self.tt_rank = actual_rank
        else:
            self.token_type_embeddings = None
        
        # Simplified encoder layers (using TransformerEncoderLayer for speed)
        self.encoder_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = nn.TransformerEncoderLayer(d_model=rank, nhead=config.num_attention_heads, batch_first=True)
            self.encoder_layers.append(layer)
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
        cls_output = embeddings[:, 0, :]
        logits = self.classifier(cls_output)
        return {"logits": logits}
import torch
import torch.nn as nn
from transformers import BertModel
from .config import SOAConfig
from .model import SOABert

def compute_global_basis_from_bert(bert_model: BertModel, k_rank: int):
    """Compute shared basis from QKV weights of a trained BERT."""
    q_weights, k_weights, v_weights = [], [], []
    for layer in bert_model.encoder.layer:
        q_weights.append(layer.attention.self.query.weight.detach())
        k_weights.append(layer.attention.self.key.weight.detach())
        v_weights.append(layer.attention.self.value.weight.detach())
    W_all = torch.cat(q_weights + k_weights + v_weights, dim=1)   # (768, 3*12*768)
    U, S, Vh = torch.linalg.svd(W_all.cpu(), full_matrices=False)
    basis = U[:, :k_rank].to(bert_model.device)
    return basis, k_rank

def migrate_bert_to_soa(bert_model: BertModel, soa_model: SOABert, basis: torch.Tensor):
    """Copy weights from BERT to SOA model using the shared basis."""
    soa_model.basis.data = basis.to(soa_model.basis.device)

    # Copy embeddings
    soa_model.word_emb.weight.data = bert_model.embeddings.word_embeddings.weight.data
    soa_model.pos_emb.weight.data = bert_model.embeddings.position_embeddings.weight.data
    soa_model.seg_emb.weight.data = bert_model.embeddings.token_type_embeddings.weight.data
    soa_model.ln.weight.data = bert_model.embeddings.LayerNorm.weight.data
    soa_model.ln.bias.data = bert_model.embeddings.LayerNorm.bias.data

    # Pooler and classifier
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        soa_model.pooler.weight.data = bert_model.pooler.dense.weight.data
        soa_model.pooler.bias.data = bert_model.pooler.dense.bias.data
    if hasattr(bert_model, 'classifier') and bert_model.classifier is not None:
        soa_model.cls.weight.data = bert_model.classifier.weight.data
        soa_model.cls.bias.data = bert_model.classifier.bias.data

    # Encoder layers: A = U^T @ W
    U = basis
    for i, (soa_layer, bert_layer) in enumerate(zip(soa_model.layers, bert_model.encoder.layer)):
        # LayerNorms
        soa_layer.ln1.weight.data = bert_layer.attention.output.LayerNorm.weight.data
        soa_layer.ln1.bias.data = bert_layer.attention.output.LayerNorm.bias.data
        soa_layer.ln2.weight.data = bert_layer.output.LayerNorm.weight.data
        soa_layer.ln2.bias.data = bert_layer.output.LayerNorm.bias.data

        # FFN
        soa_layer.ffn[0].weight.data = bert_layer.intermediate.dense.weight.data
        soa_layer.ffn[0].bias.data = bert_layer.intermediate.dense.bias.data
        soa_layer.ffn[2].weight.data = bert_layer.output.dense.weight.data
        soa_layer.ffn[2].bias.data = bert_layer.output.dense.bias.data

        # Attention projections: A = U^T @ W_orig
        W_Q = bert_layer.attention.self.query.weight.data
        W_K = bert_layer.attention.self.key.weight.data
        W_V = bert_layer.attention.self.value.weight.data
        A_Q = (U.t() @ W_Q).t()   # (H, R) -> (R, H) for nn.Linear weight
        A_K = (U.t() @ W_K).t()
        A_V = (U.t() @ W_V).t()
        soa_layer.attn.q_proj.weight.data = A_Q
        soa_layer.attn.k_proj.weight.data = A_K
        soa_layer.attn.v_proj.weight.data = A_V
        soa_layer.attn.o_proj.weight.data = bert_layer.attention.output.dense.weight.data
        soa_layer.attn.o_proj.bias.data = bert_layer.attention.output.dense.bias.data

    return soa_model
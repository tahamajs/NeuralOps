# studio/soa/svd_migration.py
import torch
import torch.nn as nn
from transformers import BertModel

def compute_global_basis(bert_model: BertModel, k_rank: int):
    qw, kw, vw = [], [], []
    for layer in bert_model.encoder.layer:
        qw.append(layer.attention.self.query.weight)
        kw.append(layer.attention.self.key.weight)
        vw.append(layer.attention.self.value.weight)
    W = torch.cat([torch.cat(qw, dim=1), torch.cat(kw, dim=1), torch.cat(vw, dim=1)], dim=1)
    U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
    return U[:, :k_rank].to(bert_model.device)

def compute_global_basis_from_hidden_states(bert_model: BertModel, k_rank: int, sample_size=64):
    device = bert_model.device
    bert_model.eval()
    all_hidden = []
    dummy_ids = torch.randint(0, 30522, (sample_size, 128)).to(device)
    dummy_mask = torch.ones_like(dummy_ids).to(device)
    dummy_types = torch.zeros_like(dummy_ids).to(device)
    with torch.no_grad():
        out = bert_model(dummy_ids, attention_mask=dummy_mask, token_type_ids=dummy_types, output_hidden_states=True)
        for hs in out.hidden_states[1:]:
            all_hidden.append(hs.reshape(-1, hs.size(-1)))
    X = torch.cat(all_hidden, dim=0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    basis = Vh[:k_rank, :].t()
    return basis

def migrate_bert_to_soa(bert_model: BertModel, soa_model, basis):
    # copy embeddings
    soa_model.word_emb.weight.data = bert_model.embeddings.word_embeddings.weight
    soa_model.pos_emb.weight.data = bert_model.embeddings.position_embeddings.weight
    soa_model.seg_emb.weight.data = bert_model.embeddings.token_type_embeddings.weight
    soa_model.ln.weight.data = bert_model.embeddings.LayerNorm.weight
    soa_model.ln.bias.data = bert_model.embeddings.LayerNorm.bias
    soa_model.basis.data = basis
    # copy pooler & classifier if needed
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        soa_model.pooler.weight.data = bert_model.pooler.dense.weight
        soa_model.pooler.bias.data = bert_model.pooler.dense.bias
    # copy layer norms and FFN
    for soa_layer, bert_layer in zip(soa_model.layers, bert_model.encoder.layer):
        soa_layer.ln1.weight.data = bert_layer.attention.output.LayerNorm.weight
        soa_layer.ln1.bias.data = bert_layer.attention.output.LayerNorm.bias
        soa_layer.ln2.weight.data = bert_layer.output.LayerNorm.weight
        soa_layer.ln2.bias.data = bert_layer.output.LayerNorm.bias
        soa_layer.ffn[0].weight.data = bert_layer.intermediate.dense.weight
        soa_layer.ffn[0].bias.data = bert_layer.intermediate.dense.bias
        soa_layer.ffn[2].weight.data = bert_layer.output.dense.weight
        soa_layer.ffn[2].bias.data = bert_layer.output.dense.bias
        # attention projections: A = U^T W
        W_q = bert_layer.attention.self.query.weight
        W_k = bert_layer.attention.self.key.weight
        W_v = bert_layer.attention.self.value.weight
        soa_layer.attn.q_proj.weight.data = (basis.t() @ W_q).t()
        soa_layer.attn.k_proj.weight.data = (basis.t() @ W_k).t()
        soa_layer.attn.v_proj.weight.data = (basis.t() @ W_v).t()
        soa_layer.attn.out_proj.weight.data = bert_layer.attention.output.dense.weight
        soa_layer.attn.out_proj.bias.data = bert_layer.attention.output.dense.bias
    return soa_model
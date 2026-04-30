# framework/soa/svd_migration.py
import torch
import torch.nn as nn
from transformers import BertModel
from .config import SOAConfig

def compute_global_basis(bert_model: BertModel, k_rank: int):
    """Compute basis from concatenated QKV weights (original method)."""
    q_weights, k_weights, v_weights = [], [], []
    for layer in bert_model.encoder.layer:
        q_weights.append(layer.attention.self.query.weight.detach().cpu())
        k_weights.append(layer.attention.self.key.weight.detach().cpu())
        v_weights.append(layer.attention.self.value.weight.detach().cpu())

    W_combined = torch.cat([torch.cat(q_weights, dim=1),
                            torch.cat(k_weights, dim=1),
                            torch.cat(v_weights, dim=1)], dim=1)
    U, S, Vh = torch.linalg.svd(W_combined, full_matrices=False)
    basis = U[:, :k_rank]
    return basis

def compute_global_basis_from_hidden_states(bert_model: BertModel, k_rank: int, sample_size=64):
    """Compute basis using hidden states from random inputs (more robust)."""
    device = next(bert_model.parameters()).device
    bert_model.eval()
    all_hidden = []

    dummy_ids = torch.randint(0, 30522, (sample_size, 128)).to(device)
    dummy_mask = torch.ones_like(dummy_ids).to(device)
    dummy_types = torch.zeros_like(dummy_ids).to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=dummy_ids, attention_mask=dummy_mask,
                             token_type_ids=dummy_types, output_hidden_states=True)
        for hs in outputs.hidden_states[1:]:
            all_hidden.append(hs.reshape(-1, hs.shape[-1]))

    X = torch.cat(all_hidden, dim=0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    basis = Vh[:k_rank, :].t()
    return basis

def compute_global_basis_from_input_states(bert_model: BertModel, k_rank: int, sample_size=64):
    """Compute basis from input tokens across all layers."""
    device = next(bert_model.parameters()).device
    bert_model.eval()
    all_inputs = []

    dummy_ids = torch.randint(0, 30522, (sample_size, 128)).to(device)
    dummy_mask = torch.ones_like(dummy_ids).to(device)
    dummy_types = torch.zeros_like(dummy_ids).to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=dummy_ids, attention_mask=dummy_mask,
                             token_type_ids=dummy_types, output_hidden_states=True)
        for hs in outputs.hidden_states:
            all_inputs.append(hs.reshape(-1, hs.shape[-1]))

    X = torch.cat(all_inputs, dim=0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    basis = Vh[:k_rank, :].t()
    return basis

def migrate_bert_to_soa(bert_model: BertModel, soa_model, basis):
    """Transfer BERT weights to SOA model using the computed basis."""
    hidden_size = soa_model.cfg.hidden_size
    k_rank = soa_model.cfg.k_rank
    assert basis.shape == (hidden_size, k_rank)

    # Embeddings
    bert_emb = bert_model.embeddings
    soa_model.word_emb.weight.data.copy_(bert_emb.word_embeddings.weight)
    soa_model.pos_emb.weight.data.copy_(bert_emb.position_embeddings.weight)
    soa_model.seg_emb.weight.data.copy_(bert_emb.token_type_embeddings.weight)
    soa_model.ln.weight.data.copy_(bert_emb.LayerNorm.weight)
    soa_model.ln.bias.data.copy_(bert_emb.LayerNorm.bias)

    # Basis
    soa_model.basis.data.copy_(basis)

    # Pooler & classifier
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        soa_model.pooler.weight.data.copy_(bert_model.pooler.dense.weight)
        soa_model.pooler.bias.data.copy_(bert_model.pooler.dense.bias)

    # Encoder layers
    U = basis
    for soa_layer, bert_layer in zip(soa_model.layers, bert_model.encoder.layer):
        # LayerNorms
        ln_attn = bert_layer.attention.output.LayerNorm
        ln_ffn = bert_layer.output.LayerNorm
        soa_layer.ln1.weight.data.copy_(ln_attn.weight)
        soa_layer.ln1.bias.data.copy_(ln_attn.bias)
        soa_layer.ln2.weight.data.copy_(ln_ffn.weight)
        soa_layer.ln2.bias.data.copy_(ln_ffn.bias)

        # FFN
        soa_layer.ffn[0].weight.data.copy_(bert_layer.intermediate.dense.weight)
        soa_layer.ffn[0].bias.data.copy_(bert_layer.intermediate.dense.bias)
        soa_layer.ffn[2].weight.data.copy_(bert_layer.output.dense.weight)
        soa_layer.ffn[2].bias.data.copy_(bert_layer.output.dense.bias)

        # Attention projections: A = U^T @ W -> then transpose for Linear weight
        W_q = bert_layer.attention.self.query.weight
        W_k = bert_layer.attention.self.key.weight
        W_v = bert_layer.attention.self.value.weight
        soa_layer.attn.q_proj.weight.data.copy_((U.t() @ W_q).t())
        soa_layer.attn.k_proj.weight.data.copy_((U.t() @ W_k).t())
        soa_layer.attn.v_proj.weight.data.copy_((U.t() @ W_v).t())
        soa_layer.attn.out_proj.weight.data.copy_(bert_layer.attention.output.dense.weight)
        soa_layer.attn.out_proj.bias.data.copy_(bert_layer.attention.output.dense.bias)

    return soa_model
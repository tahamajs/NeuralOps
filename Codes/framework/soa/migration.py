# framework/soa/migration.py
"""
Migration utilities for converting BERT to SOA format.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from .config import SOAConfig
from .model import SOABert


def compute_global_basis_from_bert(bert_model: BertModel, k_rank: int):
    """
    Compute shared basis from QKV weights of a trained BERT.
    
    This function extracts query, key, and value weights from all layers,
    concatenates them, and performs SVD to get the global basis.
    
    Args:
        bert_model: Pretrained BERT model
        k_rank: Target rank for the basis
        
    Returns:
        Tuple of (basis tensor, effective_rank)
    """
    print(f"🔍 Computing global basis from BERT (target rank={k_rank})...")
    
    q_weights, k_weights, v_weights = [], [], []
    
    for layer in bert_model.encoder.layer:
        q_weights.append(layer.attention.self.query.weight.detach())
        k_weights.append(layer.attention.self.key.weight.detach())
        v_weights.append(layer.attention.self.value.weight.detach())
    
    # Concatenate all QKV weights
    W_all = torch.cat(q_weights + k_weights + v_weights, dim=1)  # (hidden, 3 * num_layers * hidden)
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(W_all.cpu(), full_matrices=False)
    
    # Take top-k basis vectors
    effective_rank = min(k_rank, U.shape[1])
    basis = U[:, :effective_rank].to(bert_model.device)
    
    # Calculate energy retained
    energy = torch.cumsum(S[:effective_rank], dim=0) / S.sum()
    print(f"   ✅ Computed basis (rank={effective_rank}, energy={energy[-1]:.2%})")
    
    return basis, effective_rank


def migrate_bert_to_soa(bert_model: BertModel, soa_model: SOABert, basis: torch.Tensor) -> SOABert:
    """
    Copy weights from BERT to SOA model using the shared basis.
    
    This function transfers embeddings, LayerNorms, FFN weights, and projects
    attention QKV weights using the shared basis.
    
    Args:
        bert_model: Source BERT model
        soa_model: Target SOA model
        basis: Shared basis tensor (hidden, k_rank)
        
    Returns:
        SOA model with transferred weights
    """
    print("🔄 Migrating weights from BERT to SOA...")
    
    # Set shared basis
    soa_model.basis.data = basis.to(soa_model.basis.device)
    
    # --------------------------------------------------------------------
    # Copy embeddings
    # --------------------------------------------------------------------
    soa_model.word_emb.weight.data = bert_model.embeddings.word_embeddings.weight.data
    soa_model.pos_emb.weight.data = bert_model.embeddings.position_embeddings.weight.data
    soa_model.seg_emb.weight.data = bert_model.embeddings.token_type_embeddings.weight.data
    soa_model.ln.weight.data = bert_model.embeddings.LayerNorm.weight.data
    soa_model.ln.bias.data = bert_model.embeddings.LayerNorm.bias.data
    
    # --------------------------------------------------------------------
    # Copy pooler and classifier (if they exist)
    # --------------------------------------------------------------------
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        soa_model.pooler.weight.data = bert_model.pooler.dense.weight.data
        soa_model.pooler.bias.data = bert_model.pooler.dense.bias.data
    
    if hasattr(bert_model, 'classifier') and bert_model.classifier is not None:
        soa_model.cls.weight.data = bert_model.classifier.weight.data
        soa_model.cls.bias.data = bert_model.classifier.bias.data
    
    # --------------------------------------------------------------------
    # Copy encoder layers
    # --------------------------------------------------------------------
    U = basis  # (hidden, k_rank)
    
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
        
        A_Q = (U.t() @ W_Q).t()  # (k_rank, hidden) -> (hidden, k_rank)
        A_K = (U.t() @ W_K).t()
        A_V = (U.t() @ W_V).t()
        
        soa_layer.attn.q_proj.weight.data = A_Q
        soa_layer.attn.k_proj.weight.data = A_K
        soa_layer.attn.v_proj.weight.data = A_V
        
        # Output projection (no compression)
        soa_layer.attn.o_proj.weight.data = bert_layer.attention.output.dense.weight.data
        if bert_layer.attention.output.dense.bias is not None:
            soa_layer.attn.o_proj.bias.data = bert_layer.attention.output.dense.bias.data
    
    print("   ✅ Migration complete!")
    return soa_model


def quick_migrate(model_name: str = "bert-base-uncased", k_rank: int = 64) -> SOABert:
    """
    Quick migration from pretrained BERT to SOA.
    
    Args:
        model_name: HuggingFace BERT model name
        k_rank: Target rank for shared basis
        
    Returns:
        SOABert model with migrated weights
    """
    from transformers import BertModel
    
    # Load pretrained BERT
    bert_model = BertModel.from_pretrained(model_name)
    
    # Create SOA config
    cfg = SOAConfig.from_pretrained(model_name, k_rank=k_rank)
    
    # Compute basis and migrate
    basis, _ = compute_global_basis_from_bert(bert_model, k_rank)
    soa_model = SOABert(cfg)
    soa_model = migrate_bert_to_soa(bert_model, soa_model, basis)
    
    return soa_model
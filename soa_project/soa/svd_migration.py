import torch
import torch.nn as nn
from transformers import BertModel
from .config import SOAConfig

def compute_global_basis(bert_model: BertModel, k_rank: int):
    """
    Computes the global basis matrix U of shape [hidden_size, k_rank]
    by performing SVD on concatenated Q, K, V weight matrices.
    """
    print(f"  Extracting QKV weights from {len(bert_model.encoder.layer)} layers...")
    all_weights = []
    for layer in bert_model.encoder.layer:
        # Concatenate Q, K, V weights for this layer
        q_w = layer.attention.self.query.weight.data
        k_w = layer.attention.self.key.weight.data
        v_w = layer.attention.self.value.weight.data
        all_weights.extend([q_w, k_w, v_w])
    
    # Concatenate all weights horizontally to form a large matrix W_combined
    W_combined = torch.cat(all_weights, dim=1) # Shape: (H, 3 * L * H)
    
    print(f"  Performing SVD on combined weight matrix of shape {W_combined.shape}...")
    # Perform SVD. U will have shape (H, H)
    U, S, Vh = torch.linalg.svd(W_combined, full_matrices=False)
    
    # Select the top k_rank singular vectors
    basis = U[:, :k_rank]
    print(f"  Computed basis U with shape {basis.shape}")
    return basis

def migrate_bert_to_soa(bert_model: BertModel, soa_model, basis: torch.Tensor):
    """
    Migrates weights from a trained BERT model to an SOA model using the shared basis U.
    The key formula is A = U^T @ W, based on the approximation W ≈ U @ A.
    """
    cfg = soa_model.cfg
    device = next(bert_model.parameters()).device
    U = basis.to(device) # Shape: (H, R)

    print("  Migrating embeddings...")
    # 1. Migrate Embeddings
    bert_emb = bert_model.embeddings
    soa_model.word_emb.weight.data.copy_(bert_emb.word_embeddings.weight.data)
    soa_model.pos_emb.weight.data.copy_(bert_emb.position_embeddings.weight.data)
    soa_model.seg_emb.weight.data.copy_(bert_emb.token_type_embeddings.weight.data)
    soa_model.ln.weight.data.copy_(bert_emb.LayerNorm.weight.data)
    soa_model.ln.bias.data.copy_(bert_emb.LayerNorm.bias.data)

    # 2. Migrate Shared Basis
    soa_model.basis.data.copy_(U)
    print(f"  Copied shared basis U of shape {U.shape} to SOA model.")

    # 3. Migrate Pooler (if exists)
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        print("  Migrating pooler...")
        soa_model.pooler.weight.data.copy_(bert_model.pooler.dense.weight.data)
        soa_model.pooler.bias.data.copy_(bert_model.pooler.dense.bias.data)

    print("  Migrating encoder layers...")
    # 4. Migrate Encoder Layers
    for i, (soa_layer, bert_layer) in enumerate(zip(soa_model.layers, bert_model.encoder.layer)):
        print(f"    Processing layer {i+1}...")
        # LayerNorms
        soa_layer.ln1.weight.data.copy_(bert_layer.attention.output.LayerNorm.weight.data)
        soa_layer.ln1.bias.data.copy_(bert_layer.attention.output.LayerNorm.bias.data)
        soa_layer.ln2.weight.data.copy_(bert_layer.output.LayerNorm.weight.data)
        soa_layer.ln2.bias.data.copy_(bert_layer.output.LayerNorm.bias.data)
        
        # FFN
        soa_layer.ffn[0].weight.data.copy_(bert_layer.intermediate.dense.weight.data)
        soa_layer.ffn[0].bias.data.copy_(bert_layer.intermediate.dense.bias.data)
        soa_layer.ffn[2].weight.data.copy_(bert_layer.output.dense.weight.data)
        soa_layer.ffn[2].bias.data.copy_(bert_layer.output.dense.bias.data)
        
        # Attention Projections (the core SOA migration)
        # We want W ≈ U @ A. The optimal A is U^T @ W.
        W_Q = bert_layer.attention.self.query.weight.data # (H, H)
        W_K = bert_layer.attention.self.key.weight.data   # (H, H)
        W_V = bert_layer.attention.self.value.weight.data # (H, H)
        
        # Calculate A matrices: A = U^T @ W
        A_Q = U.T @ W_Q # (R, H)
        A_K = U.T @ W_K # (R, H)
        A_V = U.T @ W_V # (R, H)
        
        # Copy A matrices to SOA model. Our `q_proj` is a linear layer (R -> H),
        # so its weight matrix is (H, R). We need to copy the transpose of A.
        soa_layer.attn.q_proj.weight.data.copy_(A_Q.T)
        soa_layer.attn.k_proj.weight.data.copy_(A_K.T)
        soa_layer.attn.v_proj.weight.data.copy_(A_V.T)

    print("  Migration complete.")
    return soa_model
import torch

def svd_compress_matrix(W, rank):
    in_dim, out_dim = W.shape
    max_rank = min(in_dim, out_dim)
    
    if rank >= max_rank:
        return W
    
    try:
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        effective_rank = min(rank, len(S))
        
        U_r = U[:, :effective_rank]
        S_r = S[:effective_rank]
        Vt_r = Vt[:effective_rank, :]
        
        # Optimization: Avoid diag_embed for speed and memory efficiency
        # Compute U * S directly via broadcasting
        US = U_r * S_r.view(1, -1)
        
        # Final multiplication: (US) @ Vt
        W_compressed = US @ Vt_r
        
        return W_compressed
        
    except RuntimeError as e:
        return W

def apply_svd_to_model(model, rank):
    print(f"\n🔄 Starting High-Performance SVD Compression (Rank={rank})...")
    
    compressed_count = 0
    skipped_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.requires_grad:
                original_shape = param.shape
                
                bias_exists = False
                bias_data = None
                
                # Check if this parameter is weight of a Linear layer
                # We need to access the module to get bias
                parts = name.split('.')
                module_path = '.'.join(parts[:-1])
                
                try:
                    parent_module = dict(model.named_modules())[module_path]
                    bias_exists = parent_module.bias is not None
                    if bias_exists:
                        bias_data = parent_module.bias.data.clone()
                except KeyError:
                    pass

                try:
                    new_weight = svd_compress_matrix(param.data, rank)
                    
                    param.data.copy_(new_weight)
                    
                    if bias_exists and bias_data is not None:
                        # Update Bias using projection: b_new = U_k.T @ b_old
                        U, S, Vt = torch.linalg.svd(param.data, full_matrices=False)
                        eff_rank = min(rank, len(S))
                        U_r = U[:, :eff_rank]
                        
                        # Project bias to lower dimension
                        new_bias = U_r.T @ bias_data
                        parent_module.bias.data.copy_(new_bias)
                    
                    compressed_count += 1
                    
                    if "classifier" in name or "attention" in name:
                        print(f"✅ Compressed: {name} | {original_shape} -> {param.shape}")
                        
                except Exception as e:
                    skipped_count += 1
                    continue
            
            elif param.ndim != 2:
                pass
    
    print(f"\n📊 Summary: {compressed_count} layers compressed, {skipped_count} skipped.")
    return model
    
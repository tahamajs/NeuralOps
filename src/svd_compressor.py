# ==================== svd_compressor.py (نسخه نهایی + اصلاحات Fine-Tuning) ====================

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import gc
import traceback

# ✅ IMPORT CONFIG
from config import config


def svd_compress_matrix(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD decomposition on weight matrix W = U @ diag(S) @ Vt.
    
    Returns:
        V_proj: (in_features, rank) -> Used for input projection (x @ V)
        S: (rank,) -> Singular values
        U_proj: (out_features, rank) -> Used for output projection (@ U.T)
    """
    out_features, in_features = W.shape
    max_rank = min(in_features, out_features)
    
    # اگر رنک درخواستی بزرگتر یا مساوی حداقل بعد باشد، تجزیه لازم نیست
    if rank >= max_rank:
        return W, torch.ones(max_rank, device=W.device), torch.zeros(max_rank, device=W.device)
    
    try:
        # انجام SVD روی CPU برای پایداری بیشتر
        W_cpu = W.cpu().float()
        
        # W = U @ S @ Vh (Vh is V^T)
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        
        effective_rank = min(rank, len(S))
        
        # استخراج ماتریس‌ها با ابعاد صحیح
        # V_proj: نیاز داریم (in_features, rank) -> معادل Vh.T[:, :rank]
        V_proj = Vh[:effective_rank].T.to(W.device) 
        
        # S: مقادیر مفرد
        S_r = S[:effective_rank].to(W.device)
        
        # U_proj: نیاز داریم (out_features, rank) -> معادل U[:, :rank]
        U_proj = U[:, :effective_rank].to(W.device)
        
        return V_proj, S_r, U_proj
        
    except RuntimeError as e:
        print(f"⚠️ SVD failed on {W.shape}: {e}. Falling back to original weights.")
        # بازگشت به وزن اصلی در صورت شکست
        return W, torch.ones(1, device=W.device), torch.zeros(1, device=W.device)


class SVDLinearLayer(nn.Module):
    """
    A Linear layer that uses SVD decomposition for compression.
    
    Forward pass logic:
    y = x @ W_reconstructed
      = x @ (U @ diag(S) @ Vt)
      = (x @ Vt.T) @ diag(S) @ U.T  <-- This is mathematically correct but inefficient
    
    Optimized Efficient Path:
    1. Project input to low-rank space: temp = x @ V_proj  (where V_proj is V)
    2. Scale by singular values: scaled = temp * S
    3. Project back to output space: output = scaled @ U_proj.T (where U_proj is U)
    
    Dimensions:
    - x: (batch, seq, in_features)
    - V_proj: (in_features, rank)
    - S: (rank,)
    - U_proj: (out_features, rank)
    - Result: (batch, seq, out_features)
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        
        W = original_layer.weight.data
        has_bias = original_layer.bias is not None
        
        try:
            V_proj, S_r, U_proj = svd_compress_matrix(W, rank)
            
            # ذخیره پارامترها
            self.V_proj = nn.Parameter(V_proj.contiguous())
            self.S = nn.Parameter(S_r)
            self.U_proj = nn.Parameter(U_proj.contiguous())
            
            # حفظ بیاس اصلی
            if has_bias:
                self.register_parameter('bias', original_layer.bias)
            else:
                self.register_parameter('bias', None)
                
        except Exception as e:
            print(f"❌ Critical Error initializing SVDLayer for {original_layer.weight.shape}: {e}")
            # Fallback امن: استفاده از لایه معمولی اگر خطا رخ داد
            # (در عمل باید کل مدل را بازنشانی کرد، اما اینجا برای جلوگیری از کرش کامل، مقدار اولیه می‌گذاریم)
            self.V_proj = nn.Parameter(torch.eye(self.in_features, self.in_features)[:self.in_features, :min(rank, self.in_features)])
            self.S = nn.Parameter(torch.ones(min(rank, self.in_features)))
            self.U_proj = nn.Parameter(torch.eye(self.out_features, self.out_features)[:min(rank, self.out_features), :self.out_features].T)
            if has_bias:
                self.register_parameter('bias', original_layer.bias)
            else:
                self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, in_features)
        """
        # Step 1: Input Projection
        intermediate = x @ self.V_proj  # (B, L, in) @ (in, rank) -> (B, L, rank)
        
        # Step 2: Scaling
        scaled = intermediate * self.S.view(1, 1, -1)  # (B, L, rank) * (rank,) -> (B, L, rank)
        
        # Step 3: Output Projection
        output = scaled @ self.U_proj.T  # (B, L, rank) @ (rank, out) -> (B, L, out)
        
        # Add Bias
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, rank={self.rank}'


def compress_bert_true_svd(model: nn.Module, rank: int) -> nn.Module:
    """
    Compress BERT model using True SVD decomposition.
    Strategy: Compress ALL encoder layers, BUT KEEP the Classifier Head (last Linear) intact.
    This ensures the model retains its classification capability for fine-tuning.
    """
    print(f"\n🔄 Starting TRUE SVD Compression (Rank={rank})...")
    
    compressed_count = 0
    skipped_count = 0
    total_original_params = 0
    total_compressed_params = 0
    
    layers_to_compress = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # ⚠️ CRITICAL FIX: Skip the final classifier head and embeddings
            # We want to keep the classifier full precision so it can learn during fine-tuning
            if 'embeddings' in name or 'pooler' in name or 'classifier' in name:
                skipped_count += 1
                continue
            
            layers_to_compress.append({
                'name': name,
                'module': module
            })
    
    print(f"📊 Found {len(layers_to_compress)} Linear layers to compress (Skipping Embeddings/Pooler/Classifier)")
    
    with torch.no_grad():
        for layer_info in layers_to_compress:
            name = layer_info['name']
            original_layer = layer_info['module']
            
            original_params = original_layer.in_features * original_layer.out_features
            if original_layer.bias is not None:
                original_params += original_layer.out_features
            
            total_original_params += original_params
            
            # Create compressed layer
            compressed_layer = SVDLinearLayer(original_layer, rank)
            
            # Calculate compressed params
            # Params = (in*rank) + (rank) + (out*rank) + bias(out)
            compressed_params = rank * (original_layer.in_features + original_layer.out_features + 1)
            if compressed_layer.bias is not None:
                compressed_params += original_layer.out_features
            
            total_compressed_params += compressed_params
            
            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            setattr(parent, child_name, compressed_layer)
            compressed_count += 1
            
            # Log important layers
            if 'dense' in name or 'attention' in name:
                ratio = original_params / compressed_params if compressed_params > 0 else 1.0
                print(f"✅ Compressed: {name} | Ratio: {ratio:.2f}x")
    
    reduction_pct = ((total_original_params - total_compressed_params) / total_original_params) * 100
    
    print(f"\n📊 COMPRESSION SUMMARY")
    print(f"   Layers compressed: {compressed_count}")
    print(f"   Layers skipped (kept full): {skipped_count}")
    print(f"   Original params: {total_original_params:,}")
    print(f"   Compressed params: {total_compressed_params:,}")
    print(f"   Reduction: {reduction_pct:.2f}%")
    
    # Store metadata
    model.config.compression_info = {
        'method': 'true_svd',
        'rank': rank,
        'original_params': total_original_params,
        'compressed_params': total_compressed_params,
        'reduction_pct': reduction_pct,
        'layers_compressed': compressed_count,
        'skip_classifier': True # Important flag for loading later
    }
    
    return model


import torch
from transformers import PreTrainedModel

def svd_compress_input(inputs: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Apply SVD to input tensor and return compressed representation
    using eigenvectors as new tokens
    """
    # Reshape input to 2D matrix: [batch_size * seq_len, hidden_dim]
    original_shape = inputs.shape
    batch_size, seq_len, hidden_dim = original_shape
    inputs_2d = inputs.view(-1, hidden_dim)
    
    # Perform SVD
    U, S, Vt = torch.linalg.svd(inputs_2d, full_matrices=False)
    
    # Keep top 'rank' components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    # Apply softmax to singular values
    softmax_weights = torch.softmax(S_r, dim=0)
    
    # Create new token representations
    token_embeddings = U_r * softmax_weights.view(1, -1)
    
    # Project back to original hidden dimension
    compressed_inputs = token_embeddings @ Vt_r
    
    # Reshape to original dimensions
    return compressed_inputs.view(original_shape)

def apply_svd_to_inputs(model: PreTrainedModel, rank: int):
    """
    Modify model to apply SVD compression to inputs before processing
    """
    print(f"🔧 Applying SVD-based input compression (Rank={rank})...")
    
    # Replace forward method for input processing
    original_forward = model.forward
    
    def new_forward(input_ids=None, attention_mask=None, **kwargs):
        # Get embeddings from input ids
        embeddings = model.embeddings(input_ids=input_ids)
        
        # Apply SVD compression to embeddings
        compressed_embeddings = svd_compress_input(embeddings, rank)
        
        # Call original forward with compressed embeddings
        return original_forward(
            inputs_embeds=compressed_embeddings,
            attention_mask=attention_mask,
            **kwargs
        )
    
    model.forward = new_forward
    return model

# تابع اصلی که در main.py فراخوانی می‌شود
def compress_bert():
    # در اینجا مدل اصلی و توکنایزر را لود می‌کنیم
    from config import config
    from bert_loader import load_original_model
    
    model, _ = load_original_model()
    
    # اعمال SVD روی ورودی‌ها
    model = apply_svd_to_inputs(model, config.svd_rank)
    
    return model, None  # توکنایزر تغییر نمی‌کند

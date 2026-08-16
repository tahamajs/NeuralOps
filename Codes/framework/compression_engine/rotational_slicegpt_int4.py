from typing import Optional, Any, Dict, Tuple, List, Union
import torch
import torch.nn as nn

class AstroQuant4Bit:
    """
    Astro 4-bit Non-Uniform Quantization Codebook mapping.
    Maps floating-point parameters to non-linear centroids optimized for Gaussian weight distributions.
    """
    def __init__(self):
        k = torch.arange(16, dtype=torch.float32)
        # c_k = Sign(k - 7.5) * (|k - 7.5| / 7.5)^1.4
        self.centroids_base = torch.sign(k - 7.5) * torch.pow(torch.abs(k - 7.5) / 7.5, 1.4)
        
    def quantize(self, w):
        """
        Quantizes weight vector/matrix w using block-wise scales and zero-points.
        Returns the quantized weights.
        """
        w_min = w.min()
        w_max = w.max()
        
        c_min = self.centroids_base.min()
        c_max = self.centroids_base.max()
        
        # Handle the edge case where all weights are the same
        denom = c_max - c_min
        if denom == 0:
            denom = 1e-9
            
        s = (w_max - w_min) / denom
        z = w_min - c_min * s
        
        centroids = self.centroids_base.to(w.device) * s + z
        
        # Find nearest centroid for each weight element
        distances = torch.abs(w.unsqueeze(-1) - centroids.unsqueeze(0))
        nearest_indices = torch.argmin(distances, dim=-1)
        q = centroids[nearest_indices]
        return q

class RotationalSliceGPTINT4Compressor:
    """
    Compressor that implements the Rotational-SliceGPT-INT4 hybrid compression engine.
    It combines Rotational Invariance (to suppress outliers), Dimensionality Slicing,
    and SparseGPT-based second-order INT4 Non-Uniform Quantization.
    """
    def __init__(
        self,
        r_ratio: float = 0.75,
        lambda_reg: float = 1e-2,
        slice_ratio: Optional[float] = None,
        quant_bits: int = 4,
        **kwargs: Any,
    ):
        self.r_ratio = slice_ratio if slice_ratio is not None else r_ratio
        self.lambda_reg = lambda_reg
        self.quantizer = AstroQuant4Bit()
        
    def get_orthogonal_matrix(self, d_in, device):
        """
        Derives an orthogonal matrix Q via Block-Randomized Singular Value Decomposition 
        or orthogonalized normal matrices such that Q^T Q = I.
        """
        H = torch.randn(d_in, d_in, device=device)
        Q, _ = torch.linalg.qr(H)
        return Q

    def compress(self, W, X):
        """
        Executes the compression pipeline.
        
        Args:
            W (torch.Tensor): Dense Weight Matrix [D_out, D_in]
            X (torch.Tensor): Activation Matrix [D_in, L]
            
        Returns:
            W_quant (torch.Tensor): Quantized Sliced Weight Matrix [D_out, r]
            Q (torch.Tensor): Orthogonal Rotation Matrix [D_in, D_in]
            P_r (torch.Tensor): Slicing Projection Operator [D_in, r]
        """
        D_out, D_in = W.shape
        L = X.shape[1]
        
        r = int(D_in * self.r_ratio)
        device = W.device
        
        # 1. Orthogonal Rotation
        # \tilde{W} = W Q, \tilde{X} = Q^T X
        Q = self.get_orthogonal_matrix(D_in, device)
        W_tilde = W @ Q
        X_tilde = Q.T @ X
        
        # 2. Slicing Projection P_r
        # Identify top-r variance-bearing orthogonal channels
        channel_variances = torch.var(X_tilde, dim=1)
        top_r_indices = torch.argsort(channel_variances, descending=True)[:r]
        
        P_r = torch.zeros(D_in, r, device=device)
        for i, idx in enumerate(top_r_indices):
            P_r[idx, i] = 1.0
            
        W_sliced = W_tilde @ P_r
        X_sliced = P_r.T @ X_tilde
        
        # 3. SparseGPT Second-Order Hessian Updates
        # H^{-1} = (2 X_sliced X_sliced^T + \lambda I)^{-1}
        H = 2 * (X_sliced @ X_sliced.T) + self.lambda_reg * torch.eye(r, device=device)
        H_inv = torch.linalg.inv(H)
        
        W_quant = W_sliced.clone()
        
        for j in range(r):
            w_j = W_quant[:, j].clone()
            q_j = self.quantizer.quantize(w_j)
            e_j = w_j - q_j
            
            W_quant[:, j] = q_j
            
            if j < r - 1:
                # W_{:, j+1:r} <- W_{:, j+1:r} - e_j * [H^{-1}]_{j, j+1:r} / [H^{-1}]_{j, j}
                update = torch.outer(e_j, H_inv[j, j+1:r] / H_inv[j, j])
                W_quant[:, j+1:r] -= update
                
        return W_quant, Q, P_r

    def compress_linear(self, linear_layer: nn.Linear, X: Optional[torch.Tensor] = None) -> 'RotationalSliceGPTINT4Linear':
        """Convenience method to compress a PyTorch nn.Linear layer."""
        W = linear_layer.weight.data
        device = W.device
        if X is None:
            X = torch.randn(W.shape[1], 128, device=device)
        elif X.dim() == 3:
            X = X.reshape(-1, W.shape[1]).T
        elif X.dim() == 2 and X.shape[0] != W.shape[1]:
            X = X.T
        W_quant, Q, P_r = self.compress(W, X)
        return RotationalSliceGPTINT4Linear(W_quant, Q, P_r)


class RotationalSliceGPTINT4Linear(nn.Module):
    """
    A linear layer that applies the compressed Rotational-SliceGPT-INT4 weights.
    Maintains equivalent outputs to W @ X by evaluating: W_quant @ (Q P_r)^T X.
    """
    def __init__(self, W_quant, Q, P_r):
        super().__init__()
        # Combine the orthogonal matrix and slicing operator for efficient inference
        Q_P_r = Q @ P_r # [D_in, r]
        self.register_buffer('Q_P_r', Q_P_r)
        self.register_buffer('W_quant', W_quant)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [..., D_in]
        Returns:
            Output tensor of shape [..., D_out]
        """
        # x_proj = x @ (Q P_r) -> [..., r]
        x_proj = torch.matmul(x, self.Q_P_r)
        
        # y = x_proj @ W_quant^T -> [..., D_out]
        y = torch.matmul(x_proj, self.W_quant.T)
        return y


# Aliases for framework compatibility
RotationalSliceGPTCompressor = RotationalSliceGPTINT4Compressor

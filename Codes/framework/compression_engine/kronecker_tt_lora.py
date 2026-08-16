import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KroneckerTTLoRALayer(nn.Module):
    """
    Kronecker-TT-LoRA Hybrid Compression Engine Adapter.
    
    Implements a hybrid parameter-efficient fine-tuning (PEFT) mechanism by decomposing
    weight updates into a Kronecker product multiplied element-wise by a Tensor-Train (TT) core.
    Features dynamic rank adaptation based on singular value entropy.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        m1: int, n1: int, 
        m2: int, n2: int, 
        tt_shapes: tuple,
        tt_ranks: tuple = (1, 4, 4, 1),
        gamma: float = 1.0,
        r_min: int = 2,
        r_max: int = 32,
        entropy_threshold: float = 0.75
    ):
        """
        Args:
            base_layer (nn.Linear): The frozen base linear layer to adapt.
            m1, n1, m2, n2 (int): Dimensions for Kronecker factors A1 (m1 x n1) and A2 (m2 x n2).
                                  Must satisfy m1 * m2 = out_features, n1 * n2 = in_features.
            tt_shapes (tuple): 3-tuple (d1, d2, d3) for the Tensor-Train dimensions.
                               Must satisfy d1 * d2 * d3 = out_features * in_features.
            tt_ranks (tuple): 4-tuple (r0, r1, r2, r3) for the TT core ranks. r0 and r3 must be 1.
            gamma (float): Scaling factor for the adapter.
            r_min, r_max (int): Min and max bounds for dynamic rank adaptation.
            entropy_threshold (float): Target singular value entropy threshold.
        """
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False  # Freeze base weights
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        if m1 * m2 != self.out_features:
            raise ValueError(f"m1 * m2 ({m1 * m2}) must equal out_features ({self.out_features})")
        if n1 * n2 != self.in_features:
            raise ValueError(f"n1 * n2 ({n1 * n2}) must equal in_features ({self.in_features})")
        if tt_shapes[0] * tt_shapes[1] * tt_shapes[2] != self.out_features * self.in_features:
            raise ValueError("Product of tt_shapes must equal out_features * in_features")
        if tt_ranks[0] != 1 or tt_ranks[3] != 1:
            raise ValueError("Boundary TT ranks (r0 and r3) must be 1")
            
        self.m1, self.n1 = m1, n1
        self.m2, self.n2 = m2, n2
        self.tt_shapes = tt_shapes
        self.tt_ranks = list(tt_ranks)
        self.gamma = gamma
        self.r_min = r_min
        self.r_max = r_max
        self.entropy_threshold = entropy_threshold
        
        # Kronecker factors
        self.A1 = nn.Parameter(torch.randn(m1, n1) * math.sqrt(1.0 / n1))
        self.A2 = nn.Parameter(torch.zeros(m2, n2))  # Initialize A2 to zero for identity map initially
        
        # Tensor-Train cores
        self.G1 = nn.Parameter(torch.randn(self.tt_ranks[0], tt_shapes[0], self.tt_ranks[1]) * math.sqrt(1.0 / tt_shapes[0]))
        self.G2 = nn.Parameter(torch.randn(self.tt_ranks[1], tt_shapes[1], self.tt_ranks[2]) * math.sqrt(1.0 / tt_shapes[1]))
        self.G3 = nn.Parameter(torch.randn(self.tt_ranks[2], tt_shapes[2], self.tt_ranks[3]) * math.sqrt(1.0 / tt_shapes[2]))

    @staticmethod
    def factorize_dimensions(in_features: int, out_features: int) -> dict:
        """Find balanced factors for Kronecker and TT dimensions."""
        def factor_pair(n: int) -> tuple:
            for i in range(int(math.isqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n

        m1, m2 = factor_pair(out_features)
        n1, n2 = factor_pair(in_features)
        tt_shapes = (out_features, 1, in_features)
        return {
            "m1": m1, "n1": n1,
            "m2": m2, "n2": n2,
            "tt_shapes": tt_shapes,
        }

    @classmethod
    def from_linear(
        cls,
        base_layer: nn.Linear,
        m1: int = None,
        n1: int = None,
        m2: int = None,
        n2: int = None,
        tt_shapes: tuple = None,
        tt_ranks: tuple = (1, 4, 4, 1),
        gamma: float = 1.0,
        **kwargs
    ):
        """Create KroneckerTTLoRALayer directly from linear layer with auto-factorization."""
        dims = cls.factorize_dimensions(base_layer.in_features, base_layer.out_features)
        return cls(
            base_layer=base_layer,
            m1=m1 or dims["m1"],
            n1=n1 or dims["n1"],
            m2=m2 or dims["m2"],
            n2=n2 or dims["n2"],
            tt_shapes=tt_shapes or dims["tt_shapes"],
            tt_ranks=tt_ranks,
            gamma=gamma,
            **kwargs
        )

    def get_tt_tensor(self):
        """Constructs the dense tensor from TT cores."""
        # G1: (1, d1, r1) -> (d1, r1)
        g1 = self.G1.view(-1, self.tt_ranks[1])
        # G2: (r1, d2, r2) -> (r1, d2 * r2)
        g2 = self.G2.view(self.tt_ranks[1], -1)
        # G1 @ G2 -> (d1, d2 * r2) -> (d1 * d2, r2)
        core12 = torch.matmul(g1, g2).view(-1, self.tt_ranks[2])
        # G3: (r2, d3, 1) -> (r2, d3)
        g3 = self.G3.view(self.tt_ranks[2], -1)
        # core12 @ G3 -> (d1 * d2, d3) -> (d1 * d2 * d3,)
        tt_tensor = torch.matmul(core12, g3)
        return tt_tensor.view(self.out_features, self.in_features)

    def get_delta_w(self):
        """Computes the final adapter weight delta W."""
        # A1 ⊗ A2 using einsum
        kron = torch.einsum('ab,cd->acbd', self.A1, self.A2).reshape(self.m1 * self.m2, self.n1 * self.n2)
        
        # Hadamard product with TT tensor
        tt_tensor = self.get_tt_tensor()
        
        delta_w = self.gamma * (kron * tt_tensor)
        return delta_w

    def forward(self, x):
        """Forward pass with base weight + delta W."""
        base_out = self.base_layer(x)
        delta_w = self.get_delta_w()
        delta_out = F.linear(x, delta_w)
        return base_out + delta_out

    def adapt_tt_ranks(self, delta_r=1):
        """
        Dynamic rank adaptation based on singular value entropy of Delta W.
        Should be called periodically during fine-tuning (e.g., every epoch).
        """
        with torch.no_grad():
            delta_w = self.get_delta_w()
            
            # SVD to compute singular values
            U, S, V = torch.linalg.svd(delta_w, full_matrices=False)
            
            # Normalize singular values
            S_norm = S / (S.sum() + 1e-9)
            S_norm = S_norm[S_norm > 1e-9]
            
            # Compute singular value entropy
            h_sv = -torch.sum(S_norm * torch.log(S_norm)).item()
            
            if h_sv > self.entropy_threshold:
                direction = 1  # Increase rank to capture complexity
            elif h_sv < self.entropy_threshold:
                direction = -1 # Decrease rank to save computation/parameters
            else:
                direction = 0
                
            if direction != 0:
                new_r1 = max(self.r_min, min(self.r_max, self.tt_ranks[1] + direction * delta_r))
                new_r2 = max(self.r_min, min(self.r_max, self.tt_ranks[2] + direction * delta_r))
                
                if new_r1 != self.tt_ranks[1] or new_r2 != self.tt_ranks[2]:
                    self._resize_cores(new_r1, new_r2)
                    
    def _resize_cores(self, new_r1, new_r2):
        """Resizes the TT cores while preserving existing values and momentum."""
        old_r1, old_r2 = self.tt_ranks[1], self.tt_ranks[2]
        
        new_G1 = torch.zeros(1, self.tt_shapes[0], new_r1, device=self.G1.device, dtype=self.G1.dtype)
        new_G2 = torch.zeros(new_r1, self.tt_shapes[1], new_r2, device=self.G2.device, dtype=self.G2.dtype)
        new_G3 = torch.zeros(new_r2, self.tt_shapes[2], 1, device=self.G3.device, dtype=self.G3.dtype)
        
        # Initialize new parameters with small noise
        new_G1.normal_(0, math.sqrt(1.0 / self.tt_shapes[0]))
        new_G2.normal_(0, math.sqrt(1.0 / self.tt_shapes[1]))
        new_G3.normal_(0, math.sqrt(1.0 / self.tt_shapes[2]))
        
        # Copy old values
        min_r1 = min(old_r1, new_r1)
        min_r2 = min(old_r2, new_r2)
        
        new_G1[:, :, :min_r1] = self.G1.data[:, :, :min_r1]
        new_G2[:min_r1, :, :min_r2] = self.G2.data[:min_r1, :, :min_r2]
        new_G3[:min_r2, :, :] = self.G3.data[:min_r2, :, :]
        
        # Update parameters
        self.G1 = nn.Parameter(new_G1)
        self.G2 = nn.Parameter(new_G2)
        self.G3 = nn.Parameter(new_G3)
        self.tt_ranks[1] = new_r1
        self.tt_ranks[2] = new_r2


class KroneckerTTLoRACompressor:
    """
    Compressor wrapper to apply Kronecker-TT-LoRA to linear layers.
    """
    def __init__(self, rank: int = 8, gamma: float = 1.0, **kwargs):
        self.rank = rank
        self.gamma = gamma
        self.kwargs = kwargs

    def compress_linear(self, linear_layer: nn.Linear) -> nn.Module:
        in_f = linear_layer.in_features
        out_f = linear_layer.out_features
        
        def factor_pair(n):
            for i in range(int(math.isqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n
            
        m1, m2 = factor_pair(out_f)
        n1, n2 = factor_pair(in_f)
        
        d1 = out_f
        d2 = 1
        d3 = in_f
        
        return KroneckerTTLoRALayer(
            base_layer=linear_layer,
            m1=m1, n1=n1, m2=m2, n2=n2,
            tt_shapes=(d1, d2, d3),
            tt_ranks=(1, min(self.rank, 4), min(self.rank, 4), 1),
            gamma=self.gamma,
        )

"""
Tensor Train (TT) Decomposition Layer for PyTorch
Based on the torchTT package by ion-g-ion.
"""

import torch
import torch.nn as nn
import torchtt

class TTLinearLayer(nn.Module):
    """
    A linear layer with weights stored in Tensor Train (TT) format.
    
    This layer can be used as a drop‑in replacement for nn.Linear.
    The weight matrix W of shape (out_features, in_features) is represented
    as a TT‑tensor, dramatically reducing the number of parameters.
    """
    def __init__(self, in_features: int, out_features: int,
                 tt_rank: list, tensorized_shape_in: tuple = None,
                 tensorized_shape_out: tuple = None,
                 bias: bool = True, dtype=torch.float32):
        """
        Args:
            in_features: input dimension.
            out_features: output dimension.
            tt_rank: TT‑rank (list of ints, length = number of dimensions).
            tensorized_shape_in: factorization of in_features (e.g., (16, 48)).
            tensorized_shape_out: factorization of out_features (e.g., (16, 48)).
            bias: whether to include a bias term.
            dtype: data type of the weights.
        """
        super().__init__()
        # Determine tensorized shapes automatically if not provided
        if tensorized_shape_in is None:
            tensorized_shape_in = self._auto_tensorize(in_features)
        if tensorized_shape_out is None:
            tensorized_shape_out = self._auto_tensorize(out_features)
        
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank
        self.tensorized_shape_in = tensorized_shape_in
        self.tensorized_shape_out = tensorized_shape_out
        
        # Create the TT layer using torchtt
        self.tt_layer = torchtt.nn.LinearLayerTT(
            size_in=list(tensorized_shape_in),
            size_out=list(tensorized_shape_out),
            rank=tt_rank,
            dtype=dtype,
            initializer='He'
        )
        
        # Bias (if requested)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    @staticmethod
    def _auto_tensorize(dim: int) -> tuple:
        """
        Automatically factorize a dimension into a near‑square product.
        Example: 768 -> (16, 48) because 16 * 48 = 768.
        """
        # Find the two factors that are closest to sqrt(dim)
        import math
        a = int(math.sqrt(dim))
        while dim % a != 0:
            a -= 1
        b = dim // a
        # Ensure the first factor is the larger one (for consistency)
        if a < b:
            a, b = b, a
        return (a, b)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, tt_rank: list, **kwargs):
        """
        Initialise a TTLinearLayer from a pre‑trained nn.Linear layer.
        
        Args:
            linear: the original nn.Linear layer.
            tt_rank: TT‑rank for the decomposition.
            **kwargs: passed to TTLinearLayer.__init__.
        """
        # Create an empty TT layer
        tt_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            tt_rank=tt_rank,
            bias=linear.bias is not None,
            **kwargs
        )
        # Decompose the pre‑trained weight matrix into TT format
        weight_tensor = linear.weight.data
        # Reshape to the tensorized shape
        tensorized_shape = tt_layer.tensorized_shape_out + tt_layer.tensorized_shape_in
        weight_tensor = weight_tensor.view(tensorized_shape)
        # Decompose using torchtt (low‑rank approximation)
        tt_tensor = torchtt.tensor(weight_tensor, tt_rank)
        # Copy the cores into the TT layer
        for i, core in enumerate(tt_tensor.cores):
            tt_layer.tt_layer.cores[i].data = core.clone()
        # Copy bias if present
        if linear.bias is not None and tt_layer.bias is not None:
            tt_layer.bias.data = linear.bias.data.clone()
        return tt_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute y = Wx + b using the TT representation."""
        # The TT layer expects inputs shaped according to its tensorized dimensions.
        # We reshape the input accordingly.
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        # Perform the TT multiplication
        y = self.tt_layer(x_flat)
        # Add bias
        if self.bias is not None:
            y = y + self.bias
        # Reshape back to batch shape + out_features
        return y.view(*batch_shape, self.out_features)
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'tt_rank={self.tt_rank}, bias={self.bias is not None}')
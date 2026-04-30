import torch
import torch.nn as nn

class SVDLinearLayer(nn.Module):
    """Replace a linear layer with low‑rank approximation U * diag(S) * V."""
    def __init__(self, linear_layer: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = min(rank, self.in_features, self.out_features)

        # Perform SVD on the weight matrix
        W = linear_layer.weight.data.cpu().numpy()
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        # Keep top‑rank components
        self.U = nn.Parameter(torch.from_numpy(U[:, :self.rank]).float())
        self.S = nn.Parameter(torch.from_numpy(S[:self.rank]).float())
        self.Vh = nn.Parameter(torch.from_numpy(Vh[:self.rank, :]).float())

        # Bias
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Reconstruct weight on‑the‑fly: U @ diag(S) @ Vh
        # Efficient forward: (x @ Vh^T) @ (U * S)
        x_v = torch.matmul(x, self.Vh.t())
        sv = self.U.t() * self.S.unsqueeze(0)
        out = torch.matmul(x_v, sv.t())
        if self.bias is not None:
            out += self.bias
        return out
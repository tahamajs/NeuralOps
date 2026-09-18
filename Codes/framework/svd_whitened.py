import torch


def cholesky_whitened_svd(W, X, rank):
    """
    Cholesky-whitened SVD for model compression.

    Args:
        W: Weight matrix (out_features, in_features)
        X: Activation matrix (n_samples, in_features)
        rank: Target rank for truncation

    Returns:
        W_compressed: Truncated weight (out_features, in_features)
    """
    device = W.device
    W = W.float()
    X = X.float().to(device)

    # 1. Compute covariance
    C = (X.T @ X) / X.shape[0] + 1e-6 * torch.eye(X.shape[1], device=device)

    # 2. Cholesky decomposition: C = R^T R
    try:
        R = torch.linalg.cholesky(C).T  # shape (in_features, in_features)
    except RuntimeError:
        # Fallback to diagonal jitter if ill-conditioned
        C = C + 1e-4 * torch.eye(X.shape[1], device=device)
        R = torch.linalg.cholesky(C).T

    # 3. Whiten W
    R_inv = torch.linalg.inv(R)
    W_whitened = W @ R_inv

    # 4. SVD on whitened weight
    U, S, Vt = torch.linalg.svd(W_whitened, full_matrices=False)

    # 5. Truncate
    actual_rank = min(rank, S.shape[0])
    U_k = U[:, :actual_rank]
    S_k = S[:actual_rank]
    Vt_k = Vt[:actual_rank, :]

    # 6. Reconstruct
    W_compressed = (U_k @ torch.diag(S_k) @ Vt_k) @ R

    return W_compressed


def collect_activations(model, dataloader, num_samples=512):
    """
    Collect activations for each Linear layer.

    Returns:
        dict: {layer_name: tensor(num_samples, in_features)}
    """
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(input[0].detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    collected = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            if collected >= num_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids, attention_mask=attention_mask)
            collected += input_ids.shape[0]

    for h in hooks:
        h.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)[:num_samples]

    return activations

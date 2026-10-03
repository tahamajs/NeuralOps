Module neuralops.trainer.optimizers
===================================
Optimizers module for SOTA framework including Riemannian Stiefel Manifold Optimizers
and standard registered optimizers (AdamW, SGD, Adam, AdaBelief, Ranger, Lamb, Lars).

Functions
---------

`create_adabelief(params: Any, lr: float = 0.001, weight_decay: float = 0.0, **kwargs: Any) ‑> Any`
:   Build and return adabelief.

`create_adam(params: Any, lr: float = 0.001, weight_decay: float = 0.0, **kwargs: Any) ‑> Any`
:   Build and return adam.

`create_adamw(params: Any, lr: float = 0.001, weight_decay: float = 0.01, **kwargs: Any) ‑> Any`
:   Build and return adamw.

`create_lamb(params: Any, lr: float = 0.001, weight_decay: float = 0.01, **kwargs: Any) ‑> Any`
:   Build and return lamb.

`create_lars(params: Any, lr: float = 0.001, weight_decay: float = 0.0, **kwargs: Any) ‑> Any`
:   Build and return lars.

`create_muon(params: Any, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01, **kwargs: Any) ‑> Any`
:   Build and return Muon.

`create_muon_adamw_optimizer(model_or_params: Any, lr_muon: float = 0.02, lr_adam: float = 5e-05, weight_decay: float = 0.01, momentum: float = 0.95, ns_steps: int = 5) ‑> torch.optim.optimizer.Optimizer`
:   Splits parameters into 2D matrices (Muon) and non-2D/1D/embeddings (AdamW).
    Returns a unified MuonAdamW optimizer compatible with PyTorch GradScaler.

`create_muon_adamw_reg(params: Any, lr: float = 0.02, lr_adam: float = 5e-05, weight_decay: float = 0.01, **kwargs: Any) ‑> Any`
:   Build and return MuonAdamW hybrid optimizer.

`create_ranger(params: Any, lr: float = 0.001, weight_decay: float = 0.0, **kwargs: Any) ‑> Any`
:   Build and return ranger.

`create_riemannian_stiefel(params: Any, lr: float = 0.001, lr_stiefel: float | None = None, base_optimizer: str = 'adamw', weight_decay: float = 0.01, **kwargs: Any) ‑> neuralops.trainer.optimizers.RiemannianStiefelOptimizer`
:   Build and return RiemannianStiefelOptimizer.

`create_sgd(params: Any, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0, **kwargs: Any) ‑> Any`
:   Build and return sgd.

`is_stiefel_parameter(name: str, tensor: torch.Tensor, orthogonality_threshold: float = 0.15) ‑> bool`
:   Determines whether a parameter tensor represents an orthogonal Stiefel manifold matrix (U, V).
    
    Args:
        name: Name of the parameter (e.g. 'u_proj.weight', 'svd_u', 'basis_v').
        tensor: Parameter tensor data.
        orthogonality_threshold: Maximum deviation threshold for semi-orthogonality detection.
    
    Returns:
        bool: True if parameter should be optimized on Stiefel manifold.

`project_stiefel_gradient(W: torch.Tensor, G: torch.Tensor) ‑> torch.Tensor`
:   Projects Euclidean gradient G onto the tangent space of the Stiefel manifold St(p, n).
    
    For column-orthonormal matrices W in R^{n x p} with n >= p (W^T W = I_p):
        grad_R = G - W @ (W^T @ G + G^T @ W) / 2 = G - W @ sym(W^T G)
    
    For row-orthonormal matrices W in R^{n x p} with n < p (W W^T = I_n):
        grad_R = G - (G @ W^T + W @ G^T) / 2 @ W = G - sym(G W^T) @ W
    
    Args:
        W: Current orthogonal parameter tensor on Stiefel manifold.
        G: Euclidean gradient tensor (same shape as W).
    
    Returns:
        torch.Tensor: Riemannian gradient in tangent space T_W St(p, n).

`stiefel_qr_retraction(W: torch.Tensor, grad_R: torch.Tensor, lr: float) ‑> torch.Tensor`
:   Applies retraction step via QR decomposition to project back onto the Stiefel manifold:
        Q, R = torch.linalg.qr(W - lr * grad_R)
        W_next = Q @ diag(sign(diag(R)))
    
    Args:
        W: Current orthogonal parameter tensor.
        grad_R: Riemannian gradient in tangent space.
        lr: Learning rate / step size along tangent vector.
    
    Returns:
        torch.Tensor: Retracted orthogonal parameter tensor on Stiefel manifold.

`zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-07) ‑> torch.Tensor`
:   Newton-Schulz quintic iteration to compute the zeroth power / orthogonalization
    of a 2-D matrix: O = G (G^T G)^{-1/2}.

Classes
-------

`Muon(params: Iterable[nn.Parameter], lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01, ns_steps: int = 5)`
:   Muon - Momentum Orthogonalized Optimizer for 2D matrix parameters.
    Applies Newton-Schulz orthogonalization to momentum buffers of 2D weights.

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure: Callable[[], float] | None = None) ‑> float | None`
    :

`MuonAdamW(param_groups: list[dict[str, Any]])`
:   Hybrid optimizer: applies Muon to 2D matrix parameters and AdamW to all other parameters.
    Supports PyTorch GradScaler unscaling across both parameter groups.

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure: Callable[[], float] | None = None) ‑> float | None`
    :

`RiemannianStiefelOptimizer(params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]] | nn.Module | list[tuple[str, torch.nn.Parameter]], lr: float = 0.001, lr_stiefel: float | None = None, base_optimizer: str = 'adamw', betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, momentum: float = 0.9, stiefel_momentum: float = 0.0, dampening: float = 0.0, nesterov: bool = False, orthogonality_threshold: float = 0.15, **kwargs: Any)`
:   Riemannian Stiefel Manifold Optimizer with Retraction-based Updates.
    
    Optimizes orthogonal low-rank components (U, V) on the Stiefel manifold:
        St(p, n) = { W in R^{n x p} : W^T W = I_p }
    via tangent space gradient projection and QR retraction, while optimizing
    unconstrained Euclidean parameters (biases, layer norms, prediction heads)
    using AdamW or SGD.
    
    Args:
        params: Model parameters, iterable of parameters or param groups, or nn.Module.
        lr: Learning rate for Euclidean parameters (and default for Stiefel).
        lr_stiefel: Specific learning rate for Stiefel manifold parameters (defaults to lr).
        base_optimizer: Base optimizer for Euclidean parameters ('adamw' or 'sgd').
        betas: AdamW beta coefficients (default: (0.9, 0.999)).
        eps: AdamW epsilon for numerical stability (default: 1e-8).
        weight_decay: Weight decay applied ONLY to Euclidean parameters (default: 0.01).
        momentum: Momentum factor for SGD (default: 0.9).
        stiefel_momentum: Optional Riemannian momentum factor (default: 0.0).
        dampening: SGD dampening factor (default: 0.0).
        nesterov: Enables Nesterov momentum for SGD (default: False).
        orthogonality_threshold: Threshold for semi-orthogonality tensor auto-detection (default: 0.15).

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Static methods

    `project_gradient(W: torch.Tensor, G: torch.Tensor) ‑> torch.Tensor`
    :   Static utility for Riemannian gradient projection onto Stiefel tangent space.

    `retract(W: torch.Tensor, grad_R: torch.Tensor, lr: float) ‑> torch.Tensor`
    :   Static utility for QR retraction onto Stiefel manifold.

    ### Methods

    `check_orthogonality_error(self, param: torch.Tensor | None = None) ‑> float | dict[int, float]`
    :   Computes Frobenius norm orthogonality error ||W^T W - I||_F for manifold parameters.
        
        Args:
            param: Optional specific tensor to inspect. If None, checks all Stiefel parameters.
        
        Returns:
            float or Dict[int, float]: Orthogonality error (should be < 1e-5).

    `step(self, closure: Callable[[], float] | None = None) ‑> float | None`
    :   Performs a single Riemannian optimization step.
        
        Applies tangent space projection + QR retraction on Stiefel parameters
        and AdamW or SGD updates on Euclidean parameters.
        
        Args:
            closure: Optional evaluation closure for reevaluating the model loss.
        
        Returns:
            Optional[float]: Loss evaluated by closure if provided.
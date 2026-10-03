Module neuralops.benchmarks.run_whitened_svd
============================================

Functions
---------

`cholesky_whitened_svd(W: torch.Tensor, X: torch.Tensor, rank: int, eps: float = 0.0001) ‑> torch.Tensor`
:   Computes Cholesky-whitened SVD on weight matrix W using activation covariance X^T X.

`collect_activations(model: torch.nn.modules.module.Module, dataloader: Any, num_samples: int = 512) ‑> dict[str, torch.Tensor]`
:   Collects input activations for all Linear layers using forward hooks.

`compress_whitened(model_path, dataset_name, rank, num_samples, output_dir)`
:
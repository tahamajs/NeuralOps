Module neuralops.trainer.trainer_svd
====================================

Functions
---------

`benchmark_model(model: torch.nn.modules.module.Module, data_loader: torch.utils.data.dataloader.DataLoader, device: torch.device) ‑> dict`
:   Benchmark model.

`count_parameters(model: torch.nn.modules.module.Module) ‑> int`
:   Count parameters.

`evaluate_model(model, data_loader, device, num_labels, is_multichoice=False) ‑> Any`
:   Evaluate model.

`train_model(model: torch.nn.modules.module.Module, train_loader: torch.utils.data.dataloader.DataLoader, val_loader: torch.utils.data.dataloader.DataLoader, device: torch.device, config, num_labels: int, is_multichoice: bool = False) ‑> Any`
:   Train model.
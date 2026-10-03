Module neuralops.core.legacy_ensemble
=====================================
DEPRECATED — Legacy ensemble modules.

These classes are kept for backward compatibility only and are NOT part of
the compression pipeline. They are not related to attention or low-rank
compression and should not be used in production.

If you need a multi-model ensemble, consider:
  - Simple voting
  - Stacking with a small meta-model
  - Hugging Face's built-in ensemble utilities

Bug-fix map (2026 revision):
  LE-01  Embedding indexing with explicit long dtype.
  LE-02  sim_mat bias computed via gather (not advanced indexing).
  LE-03  to() override removed; use standard nn.Module.to().
  LE-04  NaN-safe weight normalization.
  LE-05  performance update under no_grad + explicit clamp.
  LE-06  DeprecationWarning emitted only on class instantiation.
  LE-07  Dynamic weight extraction respects trainable_dynamic.
  LE-08  model forward wrapped with fallback for missing logits.
  LE-09  model_task_ids validated for non-empty.
  LE-10  Each model's logits moved to ensemble device.
  LE-11  combine_logits validates shapes.
  LE-12  TaskAwareAttention.copy_weights silently no-op (framework hook).

Functions
---------

`run_smart_ensemble(config: dict, dataset_path: str) ‑> None`
:   Deprecated stub. Use SmartWeightedEnsemble directly in your script.

Classes
-------

`SmartWeightedEnsemble(models: list[nn.Module], num_tasks: int = 1, device: torch.device | None = None, trainable_dynamic: bool = False)`
:   Ensemble that combines multiple model predictions using various strategies:
      - simple_average
      - weighted by performance history
      - task-aware attention
      - hybrid (attention + dynamic confidence)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `device: torch.device`
    :   Compute device from the first parameter or fallback hint.

    ### Methods

    `combine_logits(self, task_id: int, all_logits: torch.Tensor, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, method: str = 'hybrid', model_task_ids: list[int] | None = None, **kwargs: Any) ‑> torch.Tensor`
    :

    `copy_weights(self, old_attn: Any) ‑> None`
    :   LE-12: framework compatibility hook. No-op.

    `forward(self, input_ids: torch.Tensor, task_id: int = 0, attention_mask: torch.Tensor | None = None, method: str = 'hybrid', model_task_ids: list[int] | None = None, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `forward_simple_average(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) ‑> torch.Tensor`
    :

    `update_performance(self, task_id: int, model_outputs: list[torch.Tensor], target: torch.Tensor, metric: str = 'accuracy') ‑> None`
    :

`TaskAwareAttention(num_tasks: Any = 1, hidden_dim: int = 64)`
:   Learns to weight model predictions based on task embeddings and a
    pairwise task similarity matrix.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   LE-12: framework compatibility hook. This module has no pretrained
        weights to transfer; silently no-op.

    `forward(self, task_id: int, all_logits: torch.Tensor, model_task_ids: list[int]) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
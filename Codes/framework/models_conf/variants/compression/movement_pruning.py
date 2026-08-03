from .base import COMPRESSOR_REGISTRY, BaseCompressor


@COMPRESSOR_REGISTRY.register("movement_pruning")
class MovementPruning(BaseCompressor):
    def apply(self, model, amount=0.3, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from neural_compressor import MovementPruning, MovementPruningConfig

            config = MovementPruningConfig(
                pruning_frequency=100, sparsity_decay_type="exp"
            )
            pruner = MovementPruning(config)
            model = pruner.prune(model)
        except ImportError:
            raise ImportError("Install neural-compressor for movement pruning")
        return model

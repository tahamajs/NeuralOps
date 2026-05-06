from .base import BaseCompressor, COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("movement_pruning")
class MovementPruning(BaseCompressor):
    def apply(self, model, amount=0.3, **kwargs):
        try:
            from neural_compressor import MovementPruningConfig, MovementPruning
            config = MovementPruningConfig(pruning_frequency=100, sparsity_decay_type="exp")
            pruner = MovementPruning(config)
            model = pruner.prune(model)
        except ImportError:
            raise ImportError("Install neural-compressor for movement pruning")
        return model
CALLBACK_REGISTRY = Registry("callback")

# Register factories for each callback
def register_default_callbacks():
    from .resources import MemoryCallback, TimeCallback, GPUStatsCallback
    from .training import EarlyStoppingCallback, CheckpointCallback
    from .monitoring import WandbCallback, TensorBoardCallback
    from .analysis import SVDAttentionCallback, GradientNormCallback

    CALLBACK_REGISTRY.register("memory")(lambda **kw: MemoryCallback(**kw))
    CALLBACK_REGISTRY.register("time")(lambda **kw: TimeCallback(**kw))
    CALLBACK_REGISTRY.register("gpu_stats")(lambda **kw: GPUStatsCallback(**kw))
    CALLBACK_REGISTRY.register("early_stopping")(lambda **kw: EarlyStoppingCallback(**kw))
    CALLBACK_REGISTRY.register("checkpoint")(lambda **kw: CheckpointCallback(**kw))
    CALLBACK_REGISTRY.register("wandb")(lambda **kw: WandbCallback(**kw))
    CALLBACK_REGISTRY.register("tensorboard")(lambda **kw: TensorBoardCallback(**kw))
    CALLBACK_REGISTRY.register("svd_attention")(lambda **kw: SVDAttentionCallback(**kw))
    CALLBACK_REGISTRY.register("gradient_norm")(lambda **kw: GradientNormCallback(**kw))

register_default_callbacks()
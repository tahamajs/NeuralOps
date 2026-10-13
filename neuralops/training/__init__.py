"""Training package initialization."""
from neuralops.training.checkpointing import checkpoint_selective, SelectiveCheckpointModule
import neuralops.trainer.trainer as trainer
import sys
sys.modules['neuralops.training.trainer'] = trainer

# Auto-wired package members
from neuralops.training import checkpointing

try:
    from neuralops.training.layer_distill import LayerAttentionDistillLoss, LayerDistillScheduler
    __all__ = ["checkpoint_selective", "SelectiveCheckpointModule", "LayerAttentionDistillLoss", "LayerDistillScheduler"]
except ImportError:
    __all__ = ["checkpoint_selective", "SelectiveCheckpointModule"]


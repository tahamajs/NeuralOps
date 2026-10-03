import torch
from neuralops.core.attention_utils import inject_attention
from transformers import AutoModelForSequenceClassification


def test_compression_stats_attached():
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model = inject_attention(model, 'asvd', rank=128)
    assert hasattr(model, 'compression_stats')
    assert 'num_compressed' in model.compression_stats
    assert model.compression_stats['num_compressed'] >= 0
    print(f"Attached stats: {model.compression_stats}")

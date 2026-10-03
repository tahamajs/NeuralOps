import framework.attentions.aegis_hybrid_attention
import pytest
import torch
import torch.nn as nn


def test_aegis_hybrid_attention_module_present():
    assert hasattr(framework.attentions, 'aegis_hybrid_attention')

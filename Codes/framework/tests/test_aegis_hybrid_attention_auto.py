import pytest
import torch
import torch.nn as nn
import framework.attentions.aegis_hybrid_attention

def test_aegis_hybrid_attention_module_present():
    assert hasattr(framework.attentions, 'aegis_hybrid_attention')

import pytest
import torch
import torch.nn as nn
import framework.attentions.moba

def test_moba_module_present():
    assert hasattr(framework.attentions, 'moba')

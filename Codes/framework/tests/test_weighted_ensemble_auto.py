import pytest
import torch
import torch.nn as nn
import framework.attentions.weighted_ensemble

def test_weighted_ensemble_module_present():
    assert hasattr(framework.attentions, 'weighted_ensemble')

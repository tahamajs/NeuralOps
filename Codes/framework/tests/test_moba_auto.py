import framework.attentions.moba
import pytest
import torch
import torch.nn as nn


def test_moba_module_present():
    assert hasattr(framework.attentions, 'moba')

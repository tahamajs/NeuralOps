import framework.attentions.svd_linear_decay
import pytest
import torch
import torch.nn as nn


def test_svd_linear_decay_module_present():
    assert hasattr(framework.attentions, 'svd_linear_decay')

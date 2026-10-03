import framework.attentions.replace
import pytest
import torch
import torch.nn as nn


def test_replace_module_present():
    assert hasattr(framework.attentions, 'replace')

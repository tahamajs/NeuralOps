import pytest
import torch
import torch.nn as nn
import framework.attentions.replace

def test_replace_module_present():
    assert hasattr(framework.attentions, 'replace')

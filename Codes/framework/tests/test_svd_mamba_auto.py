import importlib

import pytest
import torch
import torch.nn as nn


def test_svd_mamba_module_present():
    mod = importlib.import_module("framework.attentions.SVD-Mamba")
    assert mod is not None

"""Tests for neuralops.config.method_config."""

from dataclasses import asdict
import pytest

from neuralops.config import (
    DisLoRAConfig,
    FisherConfig,
    GeLoRAConfig,
    LLRDConfig,
    LoRAConfig,
    MethodConfig,
    MixoutConfig,
    R3FConfig,
    RDropConfig,
)


class TestMethodConfigs:
    """Test individual method configurations and default hyperparameters."""

    @pytest.mark.parametrize(
        "cls,field_name,default",
        [
            (LoRAConfig, "r", 8),
            (LoRAConfig, "alpha", 16),
            (RDropConfig, "alpha", 5.0),
            (R3FConfig, "epsilon", 1e-5),
            (MixoutConfig, "p", 0.9),
        ],
    )
    def test_defaults(self, cls, field_name, default):
        cfg = cls()
        assert getattr(cfg, field_name) == default

    def test_all_configs_instantiable(self):
        for cls in [
            LoRAConfig,
            RDropConfig,
            R3FConfig,
            FisherConfig,
            GeLoRAConfig,
            LLRDConfig,
            MixoutConfig,
            DisLoRAConfig,
            MethodConfig,
        ]:
            cfg = cls()
            assert cfg is not None

    def test_lora_custom_values(self):
        cfg = LoRAConfig(r=32, alpha=64, dropout=0.2)
        assert cfg.r == 32
        assert cfg.alpha == 64
        assert cfg.dropout == 0.2

    def test_method_config_subconfigs(self):
        cfg = MethodConfig()
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.rdrop, RDropConfig)

    def test_serialization(self):
        cfg = LoRAConfig(r=16, alpha=32)
        d = asdict(cfg)
        assert d["r"] == 16
        assert d["alpha"] == 32

    def test_deserialization(self):
        d = {"r": 16, "alpha": 32, "dropout": 0.1}
        cfg = LoRAConfig(**d)
        assert cfg.r == 16

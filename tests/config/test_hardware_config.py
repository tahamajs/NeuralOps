"""Tests for neuralops.config.hardware_config."""

from dataclasses import fields
import pytest

from neuralops.config import HardwareConfig


class TestHardwareConfig:
    """Test HardwareConfig dataclass defaults and overrides."""

    def test_creation(self):
        cfg = HardwareConfig()
        assert cfg is not None

    def test_defaults(self):
        cfg = HardwareConfig()
        assert isinstance(cfg.device, str)
        assert cfg.force_cpu_for_svd is True
        assert cfg.num_workers == 4

    def test_custom_device(self):
        cfg = HardwareConfig(device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_all_fields_present(self):
        cfg = HardwareConfig()
        field_names = {f.name for f in fields(cfg)}
        expected = {
            "device",
            "force_cpu_for_svd",
            "enable_fp16",
            "enable_bf16",
            "num_workers",
            "pin_memory",
            "torch_compile",
            "empty_cache_every_step",
        }
        assert expected.issubset(field_names)

import re
import os

ATTENTION_DIR = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\attentions"

def patch_aegis():
    path = os.path.join(ATTENTION_DIR, "aegis_v3.py")
    with open(path, "r") as f: content = f.read()
    
    # Change __init__ signature
    content = content.replace(
        "def __init__(\n        self,\n        model_config: Any,\n        aegis_cfg: AegisConfig,\n        pretrained_attn: Optional[nn.Module] = None,\n        bert_mode: bool = False,\n    ) -> None:",
        "def __init__(\n        self,\n        model_config: Any,\n        aegis_cfg: Optional[AegisConfig] = None,\n        pretrained_attn: Optional[nn.Module] = None,\n        bert_mode: bool = False,\n        **kwargs\n    ) -> None:"
    )
    # Inside __init__, init aegis_cfg if None
    content = content.replace(
        "self.cfg: AegisConfig = aegis_cfg",
        "self.cfg: AegisConfig = aegis_cfg if aegis_cfg is not None else AegisConfig()\n        aegis_cfg = self.cfg"
    )
    # Aegis needs pretrained_attn for zero-shot. In generic mode, it might not be passed initially, we need to bypass or handle it.
    content = content.replace(
        "if pretrained_attn is None:\n            raise ValueError(\"AegisAttention requires pretrained_attn for zero-shot initialization\")",
        "if pretrained_attn is None:\n            # Mock for registry initialization\n            class MockAttn: pass\n            pretrained_attn = MockAttn()\n            m = torch.nn.Linear(model_config.hidden_size, model_config.hidden_size)\n            pretrained_attn.q_lin = m; pretrained_attn.k_lin = m; pretrained_attn.v_lin = m; pretrained_attn.out_lin = m"
    )
    with open(path, "w") as f: f.write(content)

def patch_spectra():
    path = os.path.join(ATTENTION_DIR, "spectra_v2.py")
    if not os.path.exists(path): return
    with open(path, "r") as f: content = f.read()
    # If the class is decorated instead of the inject function:
    # Actually wait, let's just make sure ALL classes take `config, **kwargs`
    content = content.replace(
        "def __init__(\n        self,\n        model_config: Any,\n        spectra_cfg: SpectraConfig,\n        pretrained_attn: Optional[nn.Module] = None,\n        layer_idx: int = 0,\n        total_layers: int = 12,\n    ) -> None:",
        "def __init__(\n        self,\n        model_config: Any,\n        spectra_cfg: Optional[SpectraConfig] = None,\n        pretrained_attn: Optional[nn.Module] = None,\n        layer_idx: int = 0,\n        total_layers: int = 12,\n        **kwargs\n    ) -> None:"
    )
    content = content.replace(
        "self.cfg: SpectraConfig = spectra_cfg",
        "self.cfg: SpectraConfig = spectra_cfg if spectra_cfg is not None else SpectraConfig()\n        spectra_cfg = self.cfg"
    )
    content = content.replace(
        "if pretrained_attn is None:\n            raise ValueError(\"SpectraV2 requires pretrained_attn for structural injection\")",
        "if pretrained_attn is None:\n            class MockAttn: pass\n            pretrained_attn = MockAttn()\n            m = torch.nn.Linear(model_config.hidden_size, model_config.hidden_size)\n            pretrained_attn.q_lin = m; pretrained_attn.k_lin = m; pretrained_attn.v_lin = m; pretrained_attn.out_lin = m"
    )
    with open(path, "w") as f: f.write(content)

def patch_delta():
    path = os.path.join(ATTENTION_DIR, "gated_delta_net.py")
    if not os.path.exists(path): return
    with open(path, "r") as f: content = f.read()
    content = content.replace(
        "def __init__(\n        self,\n        model_config: Any,\n        delta_cfg: DeltaNetConfig,\n    ) -> None:",
        "def __init__(\n        self,\n        model_config: Any,\n        delta_cfg: Optional[DeltaNetConfig] = None,\n        **kwargs\n    ) -> None:"
    )
    content = content.replace(
        "self.cfg: DeltaNetConfig = delta_cfg",
        "self.cfg: DeltaNetConfig = delta_cfg if delta_cfg is not None else DeltaNetConfig()"
    )
    with open(path, "w") as f: f.write(content)

patch_aegis()
patch_spectra()
patch_delta()
print("Patched models!")

import sys
from pathlib import Path

path = Path("Codes/framework/core/snapshot_restore.py")
content = path.read_text(encoding="utf-8")

if "m.merged = False" not in content:
    patch = """
        # Clean any compression/evaluation attributes attached dynamically
        for attr in ("compression_stats", "is_custom_attention", "_injected_attention"):
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except Exception:
                    pass
        
        # Reset LoRA merge state
        for m in self.model.modules():
            if hasattr(m, "merged"):
                m.merged = False
"""
    content = content.replace("""
        # Clean any compression/evaluation attributes attached dynamically
        for attr in ("compression_stats", "is_custom_attention", "_injected_attention"):
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except Exception:
                    pass
""", patch)
    path.write_text(content, encoding="utf-8")
    print("Patched snapshot_restore.py")
else:
    print("Already patched")

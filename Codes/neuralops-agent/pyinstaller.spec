# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the NeuralOps GPU Provider Agent."""
import sys
from pathlib import Path

block_cipher = None

_here = Path(__file__).parent

a = Analysis(
    ['agent/main.py'],
    pathex=[str(_here)],
    binaries=[],
    datas=[
        ('agent-config.yaml', '.'),
    ],
    hiddenimports=[
        'pynvml', 'docker', 'websocket', 'psutil',
        'speedtest', 'yaml', 'packaging', 'requests',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'notebook', 'jupyter',
        'IPython', 'sqlalchemy', 'pandas',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='neuralops-agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_here / 'resources' / 'icon.ico') if (sys.platform == 'win32') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='neuralops-agent',
)
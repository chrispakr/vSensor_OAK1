# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports_loguru = collect_submodules('loguru')
#hidden_imports_h5py = collect_submodules('h5py')
#all_hidden_imports = hiddenimports_tensorflow + hidden_imports_h5py

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('*.dt', '.')
    ],
    hiddenimports=hiddenimports_loguru,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='vSensorM4',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vSensorM4',
)

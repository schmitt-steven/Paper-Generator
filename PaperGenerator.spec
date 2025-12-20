# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Paper Generator
Build with: pyinstaller PaperGenerator.spec
"""

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

# Collect all files from sv_ttk (Sun Valley theme)
sv_ttk_datas, sv_ttk_binaries, sv_ttk_hiddenimports = collect_all('sv_ttk')

# Collect pymupdf4llm data files
pymupdf4llm_datas = collect_data_files('pymupdf4llm')

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=sv_ttk_binaries,
    datas=[
        # App data files
        ('latex_template', 'latex_template'),
        ('user_files', 'user_files'),
        ('user_settings.json', '.'),
        # Theme files
        *sv_ttk_datas,
        *pymupdf4llm_datas,
    ],
    hiddenimports=[
        # LM Studio
        'lmstudio',
        # PDF processing
        'pymupdf',
        'pymupdf4llm',
        # HTTP clients (used by lmstudio)
        'httpx',
        'httpcore',
        'anyio',
        'sniffio',
        'h11',
        'certifi',
        # Async
        'asyncio',
        # Other potential hidden imports
        'numpy',
        'json',
        'pathlib',
        *sv_ttk_hiddenimports,
    ],
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
    name='PaperGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Uncomment and add your icon file if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PaperGenerator',
)

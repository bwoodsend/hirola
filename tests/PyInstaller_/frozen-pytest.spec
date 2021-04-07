# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import pkg_resources
from PyInstaller.utils.hooks import copy_metadata

# Each pytest plugin needs to be explicitly collected.
pytest_entry_points = list(pkg_resources.iter_entry_points("pytest11"))
datas = sum((copy_metadata(i.dist.project_name)
             for i in pytest_entry_points), [])
pytest_plugins = [i.module_name for i in pytest_entry_points]

a = Analysis(['frozen-pytest.py'],
             pathex=[SPECPATH],
             binaries=[],
             datas=datas,
             hiddenimports=pytest_plugins + ["statistics"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='frozen-pytest',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               upx_exclude=[],
               name='frozen-pytest')
